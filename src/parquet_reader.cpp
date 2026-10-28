#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL AK_PARQUET_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <arrow/util/float16.h>
#include <parquet/arrow/reader.h>
#include <parquet/file_reader.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace {

struct PythonError {};

class PyRef {
public:
    explicit PyRef(PyObject* value = nullptr) : value_(value) {}
    ~PyRef() { Py_XDECREF(value_); }
    PyRef(const PyRef&) = delete;
    PyRef& operator=(const PyRef&) = delete;
    PyObject* get() const { return value_; }
    PyObject* release() { PyObject* value = value_; value_ = nullptr; return value; }
private:
    PyObject* value_;
};

class AcquireGIL {
public:
    AcquireGIL() : state_(PyGILState_Ensure()) {}
    ~AcquireGIL() { PyGILState_Release(state_); }
private:
    PyGILState_STATE state_;
};

class ReleaseGIL {
public:
    ReleaseGIL() : state_(PyEval_SaveThread()) {}
    ~ReleaseGIL() { PyEval_RestoreThread(state_); }
private:
    PyThreadState* state_;
};

class RecursionGuard {
public:
    RecursionGuard() {
        if (Py_EnterRecursiveCall(" while converting a Parquet value"))
            throw PythonError();
    }
    ~RecursionGuard() { Py_LeaveRecursiveCall(); }
};

PyObject* checked(PyObject* value) {
    if (!value) throw PythonError();
    return value;
}

void check_status(const arrow::Status& status) {
    if (status.ok()) return;
    PyObject* exception = PyExc_ValueError;
    if (status.IsIOError()) exception = PyExc_OSError;
    else if (status.IsNotImplemented()) exception = PyExc_NotImplementedError;
    else if (status.IsOutOfMemory()) exception = PyExc_MemoryError;
    else if (status.IsTypeError()) exception = PyExc_TypeError;
    PyErr_SetString(exception, status.ToString().c_str());
    throw PythonError();
}

template <typename T>
T unwrap(arrow::Result<T> result) {
    check_status(result.status());
    return std::move(result).ValueOrDie();
}

// Each operation holds the mutex across seek/read, and acquires the GIL only
// for Python callbacks. Arrow reading itself runs with the GIL released.
class PythonFile final : public arrow::io::RandomAccessFile {
public:
    explicit PythonFile(PyObject* file) : file_(Py_NewRef(file)) {}
    ~PythonFile() override {
        AcquireGIL gil;
        Py_DECREF(file_);
        if (error_type_ && !PyErr_Occurred()) {
            PyErr_Restore(error_type_, error_value_, error_traceback_);
            error_type_ = error_value_ = error_traceback_ = nullptr;
        }
        Py_XDECREF(error_type_);
        Py_XDECREF(error_value_);
        Py_XDECREF(error_traceback_);
    }

    arrow::Status Close() override { return arrow::Status::OK(); }
    bool closed() const override { return false; }

    arrow::Result<int64_t> Tell() const override {
        std::lock_guard<std::mutex> lock(mutex_);
        AcquireGIL gil;
        return tell();
    }

    arrow::Status Seek(int64_t position) override {
        std::lock_guard<std::mutex> lock(mutex_);
        AcquireGIL gil;
        return seek(position, 0);
    }

    arrow::Result<int64_t> GetSize() override {
        std::lock_guard<std::mutex> lock(mutex_);
        AcquireGIL gil;
        auto original = tell();
        if (!original.ok()) return original.status();
        auto status = seek(0, 2);
        if (!status.ok()) return status;
        auto size = tell();
        status = seek(*original, 0);
        if (!status.ok()) return status;
        return size;
    }

    arrow::Result<int64_t> Read(int64_t count, void* out) override {
        std::lock_guard<std::mutex> lock(mutex_);
        AcquireGIL gil;
        return read(count, out);
    }

    arrow::Result<std::shared_ptr<arrow::Buffer>> Read(int64_t count) override {
        auto buffer_result = arrow::AllocateResizableBuffer(count);
        if (!buffer_result.ok()) return buffer_result.status();
        std::shared_ptr<arrow::ResizableBuffer> buffer = std::move(*buffer_result);
        auto result = Read(count, buffer->mutable_data());
        if (!result.ok()) return result.status();
        auto status = buffer->Resize(*result, false);
        if (!status.ok()) return status;
        return buffer;
    }

    arrow::Result<int64_t> ReadAt(int64_t position, int64_t count, void* out) override {
        std::lock_guard<std::mutex> lock(mutex_);
        AcquireGIL gil;
        auto status = seek(position, 0);
        if (!status.ok()) return status;
        return read(count, out);
    }

    arrow::Result<std::shared_ptr<arrow::Buffer>> ReadAt(
            int64_t position, int64_t count) override {
        auto buffer_result = arrow::AllocateResizableBuffer(count);
        if (!buffer_result.ok()) return buffer_result.status();
        std::shared_ptr<arrow::ResizableBuffer> buffer = std::move(*buffer_result);
        auto result = ReadAt(position, count, buffer->mutable_data());
        if (!result.ok()) return result.status();
        auto status = buffer->Resize(*result, false);
        if (!status.ok()) return status;
        return buffer;
    }

    bool restore_error() {
        if (!error_type_) return false;
        PyErr_Restore(error_type_, error_value_, error_traceback_);
        error_type_ = error_value_ = error_traceback_ = nullptr;
        return true;
    }

private:
    arrow::Status failure() const {
        if (!error_type_) PyErr_Fetch(&error_type_, &error_value_, &error_traceback_);
        else PyErr_Clear();
        return arrow::Status::IOError("Python file operation failed");
    }

    arrow::Result<int64_t> tell() const {
        PyRef value(PyObject_CallMethod(file_, "tell", nullptr));
        if (!value.get()) return failure();
        int64_t position = PyLong_AsLongLong(value.get());
        if (PyErr_Occurred()) return failure();
        if (position < 0) return arrow::Status::IOError("negative file position");
        return position;
    }

    arrow::Status seek(int64_t position, int whence) const {
        PyRef value(PyObject_CallMethod(file_, "seek", "Li",
                static_cast<long long>(position), whence));
        if (!value.get()) return failure();
        return arrow::Status::OK();
    }

    arrow::Result<int64_t> read(int64_t count, void* out) const {
        if (count < 0 || count > PY_SSIZE_T_MAX)
            return arrow::Status::Invalid("invalid read size");
        int64_t total = 0;
        while (total < count) {
            PyRef value(PyObject_CallMethod(file_, "read", "n",
                    static_cast<Py_ssize_t>(count - total)));
            if (!value.get()) return failure();
            if (!PyBytes_Check(value.get())) {
                PyErr_SetString(PyExc_TypeError, "file.read() must return bytes");
                return failure();
            }
            Py_ssize_t size = PyBytes_GET_SIZE(value.get());
            if (size > count - total) {
                PyErr_SetString(PyExc_ValueError, "file.read() returned too many bytes");
                return failure();
            }
            if (!size) break;
            std::memcpy(static_cast<char*>(out) + total,
                    PyBytes_AS_STRING(value.get()), static_cast<size_t>(size));
            total += size;
        }
        return total;
    }

    PyObject* file_;
    mutable std::mutex mutex_;
    mutable PyObject* error_type_ = nullptr;
    mutable PyObject* error_value_ = nullptr;
    mutable PyObject* error_traceback_ = nullptr;
};

PyObject* convert_array(const std::shared_ptr<arrow::Array>& array);

PyObject* object_value(const std::shared_ptr<arrow::Array>& array, int64_t index) {
    if (array->IsNull(index)) return Py_NewRef(Py_None);
    RecursionGuard recursion;
    switch (array->type_id()) {
#define AK_SIGNED(ID, CLASS) \
        case arrow::Type::ID: return PyLong_FromLongLong( \
                static_cast<const arrow::CLASS&>(*array).Value(index));
#define AK_UNSIGNED(ID, CLASS) \
        case arrow::Type::ID: return PyLong_FromUnsignedLongLong( \
                static_cast<const arrow::CLASS&>(*array).Value(index));
        AK_SIGNED(INT8, Int8Array)
        AK_SIGNED(INT16, Int16Array)
        AK_SIGNED(INT32, Int32Array)
        AK_SIGNED(INT64, Int64Array)
        AK_UNSIGNED(UINT8, UInt8Array)
        AK_UNSIGNED(UINT16, UInt16Array)
        AK_UNSIGNED(UINT32, UInt32Array)
        AK_UNSIGNED(UINT64, UInt64Array)
#undef AK_SIGNED
#undef AK_UNSIGNED
        case arrow::Type::BOOL:
            return PyBool_FromLong(static_cast<const arrow::BooleanArray&>(*array).Value(index));
        case arrow::Type::HALF_FLOAT:
            return PyFloat_FromDouble(arrow::util::Float16::FromBits(
                    static_cast<const arrow::HalfFloatArray&>(*array).Value(index)).ToDouble());
        case arrow::Type::FLOAT:
            return PyFloat_FromDouble(static_cast<const arrow::FloatArray&>(*array).Value(index));
        case arrow::Type::DOUBLE:
            return PyFloat_FromDouble(static_cast<const arrow::DoubleArray&>(*array).Value(index));
        case arrow::Type::STRING: {
            auto value = static_cast<const arrow::StringArray&>(*array).GetView(index);
            return PyUnicode_DecodeUTF8(value.data(), value.size(), "strict");
        }
        case arrow::Type::LARGE_STRING: {
            auto value = static_cast<const arrow::LargeStringArray&>(*array).GetView(index);
            return PyUnicode_DecodeUTF8(value.data(), value.size(), "strict");
        }
        case arrow::Type::BINARY: {
            auto value = static_cast<const arrow::BinaryArray&>(*array).GetView(index);
            return PyBytes_FromStringAndSize(value.data(), value.size());
        }
        case arrow::Type::LARGE_BINARY: {
            auto value = static_cast<const arrow::LargeBinaryArray&>(*array).GetView(index);
            return PyBytes_FromStringAndSize(value.data(), value.size());
        }
        case arrow::Type::FIXED_SIZE_BINARY: {
            auto value = static_cast<const arrow::FixedSizeBinaryArray&>(*array).GetView(index);
            return PyBytes_FromStringAndSize(value.data(), value.size());
        }
        case arrow::Type::STRUCT: {
            const auto& values = static_cast<const arrow::StructArray&>(*array);
            PyRef result(checked(PyDict_New()));
            for (int field = 0; field < values.num_fields(); ++field) {
                const auto& name = values.struct_type()->field(field)->name();
                PyRef key(checked(PyUnicode_DecodeUTF8(name.data(), name.size(), "strict")));
                PyRef value(checked(object_value(values.field(field), index)));
                if (PyDict_SetItem(result.get(), key.get(), value.get()) < 0)
                    throw PythonError();
            }
            return result.release();
        }
        case arrow::Type::LIST: {
            const auto& values = static_cast<const arrow::ListArray&>(*array);
            return convert_array(values.value_slice(index));
        }
        case arrow::Type::LARGE_LIST: {
            const auto& values = static_cast<const arrow::LargeListArray&>(*array);
            return convert_array(values.value_slice(index));
        }
        case arrow::Type::FIXED_SIZE_LIST: {
            const auto& values = static_cast<const arrow::FixedSizeListArray&>(*array);
            return convert_array(values.value_slice(index));
        }
        case arrow::Type::DICTIONARY: {
            const auto& values = static_cast<const arrow::DictionaryArray&>(*array);
            return object_value(values.dictionary(), values.GetValueIndex(index));
        }
        case arrow::Type::DECIMAL128:
        case arrow::Type::DECIMAL256:
        case arrow::Type::DECIMAL32:
        case arrow::Type::DECIMAL64: {
            auto scalar = unwrap(array->GetScalar(index));
            std::string text = scalar->ToString();
            PyRef module(checked(PyImport_ImportModule("decimal")));
            PyRef cls(checked(PyObject_GetAttrString(module.get(), "Decimal")));
            PyRef value(checked(PyUnicode_FromStringAndSize(text.data(), text.size())));
            return PyObject_CallOneArg(cls.get(), value.get());
        }
        case arrow::Type::TIMESTAMP:
        case arrow::Type::DATE32:
        case arrow::Type::DATE64: {
            PyRef values(checked(convert_array(array->Slice(index, 1))));
            return PyArray_GETITEM(reinterpret_cast<PyArrayObject*>(values.get()),
                    static_cast<char*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(values.get()))));
        }
        case arrow::Type::NA:
            return Py_NewRef(Py_None);
        default:
            PyErr_Format(PyExc_NotImplementedError, "unsupported Arrow type: %s",
                    array->type()->ToString().c_str());
            throw PythonError();
    }
}

int numpy_type(const arrow::DataType& type, bool nullable) {
    switch (type.id()) {
        case arrow::Type::INT8: return nullable ? NPY_FLOAT64 : NPY_INT8;
        case arrow::Type::INT16: return nullable ? NPY_FLOAT64 : NPY_INT16;
        case arrow::Type::INT32: return nullable ? NPY_FLOAT64 : NPY_INT32;
        case arrow::Type::INT64: return nullable ? NPY_FLOAT64 : NPY_INT64;
        case arrow::Type::UINT8: return nullable ? NPY_FLOAT64 : NPY_UINT8;
        case arrow::Type::UINT16: return nullable ? NPY_FLOAT64 : NPY_UINT16;
        case arrow::Type::UINT32: return nullable ? NPY_FLOAT64 : NPY_UINT32;
        case arrow::Type::UINT64: return nullable ? NPY_FLOAT64 : NPY_UINT64;
        case arrow::Type::HALF_FLOAT: return NPY_FLOAT16;
        case arrow::Type::FLOAT: return NPY_FLOAT32;
        case arrow::Type::DOUBLE: return NPY_FLOAT64;
        case arrow::Type::BOOL: return nullable ? NPY_OBJECT : NPY_BOOL;
        case arrow::Type::TIMESTAMP:
        case arrow::Type::DATE32:
        case arrow::Type::DATE64: return NPY_DATETIME;
        case arrow::Type::NA:
        case arrow::Type::STRING:
        case arrow::Type::LARGE_STRING:
        case arrow::Type::BINARY:
        case arrow::Type::LARGE_BINARY:
        case arrow::Type::FIXED_SIZE_BINARY:
        case arrow::Type::STRUCT:
        case arrow::Type::LIST:
        case arrow::Type::LARGE_LIST:
        case arrow::Type::FIXED_SIZE_LIST:
        case arrow::Type::DICTIONARY:
        case arrow::Type::DECIMAL32:
        case arrow::Type::DECIMAL64:
        case arrow::Type::DECIMAL128:
        case arrow::Type::DECIMAL256: return NPY_OBJECT;
        default:
            PyErr_Format(PyExc_NotImplementedError, "unsupported Arrow type: %s",
                    type.ToString().c_str());
            throw PythonError();
    }
}

PyArray_Descr* datetime_dtype(const arrow::DataType& type) {
    const char* unit = "D";
    if (type.id() == arrow::Type::DATE64) unit = "ms";
    else if (type.id() == arrow::Type::TIMESTAMP) {
        switch (static_cast<const arrow::TimestampType&>(type).unit()) {
            case arrow::TimeUnit::SECOND: unit = "s"; break;
            case arrow::TimeUnit::MILLI: unit = "ms"; break;
            case arrow::TimeUnit::MICRO: unit = "us"; break;
            case arrow::TimeUnit::NANO: unit = "ns"; break;
        }
    }
    std::string spec = std::string("datetime64[") + unit + "]";
    PyRef value(checked(PyUnicode_FromString(spec.c_str())));
    PyArray_Descr* result = nullptr;
    if (!PyArray_DescrConverter(value.get(), &result)) throw PythonError();
    return result;
}

void copy_chunk(const std::shared_ptr<arrow::Array>& array,
        PyArrayObject* output, int64_t offset, int dtype) {
    if (array->length() == 0) return;
    char* data = static_cast<char*>(PyArray_DATA(output));
    if (dtype == NPY_OBJECT) {
        auto values = reinterpret_cast<PyObject**>(data);
        for (int64_t i = 0; i < array->length(); ++i) {
            PyObject* value = checked(object_value(array, i));
            Py_XDECREF(values[offset + i]);
            values[offset + i] = value;
        }
        return;
    }
    switch (array->type_id()) {
#define AK_COPY(ID, CLASS, CTYPE) \
        case arrow::Type::ID: { \
            const auto& values = static_cast<const arrow::CLASS&>(*array); \
            if (dtype == NPY_FLOAT64) { \
                auto target = reinterpret_cast<double*>(data) + offset; \
                for (int64_t i = 0; i < array->length(); ++i) \
                    target[i] = array->IsNull(i) ? Py_NAN : static_cast<double>(values.Value(i)); \
            } else { \
                std::memcpy(data + offset * sizeof(CTYPE), values.raw_values(), \
                        static_cast<size_t>(array->length()) * sizeof(CTYPE)); \
            } \
            break; \
        }
        AK_COPY(INT8, Int8Array, int8_t)
        AK_COPY(INT16, Int16Array, int16_t)
        AK_COPY(INT32, Int32Array, int32_t)
        AK_COPY(INT64, Int64Array, int64_t)
        AK_COPY(UINT8, UInt8Array, uint8_t)
        AK_COPY(UINT16, UInt16Array, uint16_t)
        AK_COPY(UINT32, UInt32Array, uint32_t)
        AK_COPY(UINT64, UInt64Array, uint64_t)
#undef AK_COPY
        case arrow::Type::HALF_FLOAT: {
            const auto& values = static_cast<const arrow::HalfFloatArray&>(*array);
            auto target = reinterpret_cast<uint16_t*>(data) + offset;
            for (int64_t i = 0; i < array->length(); ++i)
                target[i] = array->IsNull(i) ? 0x7e00 : values.Value(i);
            break;
        }
        case arrow::Type::FLOAT: {
            const auto& values = static_cast<const arrow::FloatArray&>(*array);
            auto target = reinterpret_cast<float*>(data) + offset;
            for (int64_t i = 0; i < array->length(); ++i)
                target[i] = array->IsNull(i) ? Py_NAN : values.Value(i);
            break;
        }
        case arrow::Type::DOUBLE: {
            const auto& values = static_cast<const arrow::DoubleArray&>(*array);
            auto target = reinterpret_cast<double*>(data) + offset;
            for (int64_t i = 0; i < array->length(); ++i)
                target[i] = array->IsNull(i) ? Py_NAN : values.Value(i);
            break;
        }
        case arrow::Type::BOOL: {
            const auto& values = static_cast<const arrow::BooleanArray&>(*array);
            auto target = reinterpret_cast<npy_bool*>(data) + offset;
            for (int64_t i = 0; i < array->length(); ++i) target[i] = values.Value(i);
            break;
        }
        case arrow::Type::DATE32: {
            const auto& values = static_cast<const arrow::Date32Array&>(*array);
            auto target = reinterpret_cast<int64_t*>(data) + offset;
            for (int64_t i = 0; i < array->length(); ++i)
                target[i] = array->IsNull(i) ? NPY_DATETIME_NAT : values.Value(i);
            break;
        }
        case arrow::Type::DATE64:
        case arrow::Type::TIMESTAMP: {
            auto target = reinterpret_cast<int64_t*>(data) + offset;
            auto source = array->data()->GetValues<int64_t>(1);
            for (int64_t i = 0; i < array->length(); ++i)
                target[i] = array->IsNull(i) ? NPY_DATETIME_NAT : source[i];
            break;
        }
        default:
            PyErr_SetString(PyExc_NotImplementedError, "unsupported numeric Arrow type");
            throw PythonError();
    }
}

PyObject* convert_column(const arrow::ChunkedArray& column) {
    int dtype = numpy_type(*column.type(), column.null_count() != 0);
    if (column.length() > NPY_MAX_INTP) {
        PyErr_SetString(PyExc_OverflowError, "column is too large for NumPy");
        throw PythonError();
    }
    npy_intp size = static_cast<npy_intp>(column.length());
    PyArray_Descr* descriptor = dtype == NPY_DATETIME
            ? datetime_dtype(*column.type()) : PyArray_DescrFromType(dtype);
    if (!descriptor) throw PythonError();
    PyRef result(checked(PyArray_NewFromDescr(&PyArray_Type, descriptor, 1, &size,
            nullptr, nullptr, 0, nullptr)));
    auto output = reinterpret_cast<PyArrayObject*>(result.get());
    int64_t offset = 0;
    for (const auto& chunk : column.chunks()) {
        copy_chunk(chunk, output, offset, dtype);
        offset += chunk->length();
    }
    PyArray_CLEARFLAGS(output, NPY_ARRAY_WRITEABLE);
    return result.release();
}

PyObject* convert_array(const std::shared_ptr<arrow::Array>& array) {
    return convert_column(arrow::ChunkedArray(array));
}

PyObject* read_parquet(PyObject*, PyObject* args, PyObject* kwargs) {
    try {
        PyObject* source;
        PyObject* selection = Py_None;
        static const char* keywords[] = {"file", "columns_select", nullptr};
        if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|$O:parquet_to_arrays",
                const_cast<char**>(keywords), &source, &selection)) return nullptr;

        std::vector<std::string> selected;
        if (selection != Py_None) {
            if (!PyList_Check(selection)) {
                PyErr_SetString(PyExc_TypeError, "columns_select must be a list of strings or None");
                throw PythonError();
            }
            std::unordered_set<std::string> seen;
            for (Py_ssize_t i = 0; i < PyList_GET_SIZE(selection); ++i) {
                PyObject* name = PyList_GET_ITEM(selection, i);
                if (!PyUnicode_Check(name)) {
                    PyErr_SetString(PyExc_TypeError, "columns_select must contain only strings");
                    throw PythonError();
                }
                Py_ssize_t length;
                const char* text = PyUnicode_AsUTF8AndSize(name, &length);
                if (!text) throw PythonError();
                std::string value(text, length);
                if (!seen.insert(value).second) {
                    PyErr_Format(PyExc_ValueError, "duplicate column selection: %R", name);
                    throw PythonError();
                }
                selected.push_back(std::move(value));
            }
        }

        std::shared_ptr<arrow::io::RandomAccessFile> file;
        std::shared_ptr<PythonFile> python_file;
        PyRef path(PyOS_FSPath(source));
        if (path.get()) {
            PyRef encoded(PyUnicode_Check(path.get())
                    ? PyUnicode_EncodeFSDefault(path.get()) : Py_NewRef(path.get()));
            checked(encoded.get());
            char* data;
            Py_ssize_t size;
            if (PyBytes_AsStringAndSize(encoded.get(), &data, &size) < 0) throw PythonError();
            if (std::memchr(data, '\0', size)) {
                PyErr_SetString(PyExc_ValueError, "embedded null byte in path");
                throw PythonError();
            }
            arrow::Result<std::shared_ptr<arrow::io::ReadableFile>> opened;
            {
                ReleaseGIL released;
                opened = arrow::io::ReadableFile::Open(std::string(data, size));
            }
            file = unwrap(std::move(opened));
        } else {
            if (!PyErr_ExceptionMatches(PyExc_TypeError)) throw PythonError();
            PyErr_Clear();
            for (const char* name : {"read", "seek", "tell"}) {
                PyRef method(PyObject_GetAttrString(source, name));
                if (!method.get()) {
                    PyErr_Clear();
                    PyErr_SetString(PyExc_TypeError, "file must be a path or seekable binary file");
                    throw PythonError();
                }
                if (!PyCallable_Check(method.get())) {
                    PyErr_SetString(PyExc_TypeError, "file read, seek and tell must be callable");
                    throw PythonError();
                }
            }
            python_file = std::make_shared<PythonFile>(source);
            file = python_file;
        }

        arrow::Result<std::unique_ptr<parquet::arrow::FileReader>> opened;
        {
            ReleaseGIL released;
            opened = parquet::arrow::OpenFile(file, arrow::default_memory_pool());
        }
        if (python_file && python_file->restore_error()) throw PythonError();
        auto reader = unwrap(std::move(opened));
        reader->set_use_threads(!python_file);
        std::shared_ptr<arrow::Schema> schema;
        check_status(reader->GetSchema(&schema));

        std::vector<int> fields;
        std::vector<int> leaves;
        if (selection == Py_None) {
            for (int i = 0; i < schema->num_fields(); ++i) fields.push_back(i);
        } else {
            for (const auto& name : selected) {
                auto matches = schema->GetAllFieldIndices(name);
                if (matches.size() != 1) {
                    PyErr_SetString(PyExc_ValueError, "column name is missing or ambiguous");
                    throw PythonError();
                }
                fields.push_back(matches[0]);
            }
        }
        const auto* parquet_schema = reader->parquet_reader()->metadata()->schema();
        for (int field : fields) {
            const auto* root = parquet_schema->group_node()->field(field).get();
            for (int leaf = 0; leaf < parquet_schema->num_columns(); ++leaf) {
                if (parquet_schema->GetColumnRoot(leaf) == root) leaves.push_back(leaf);
            }
        }
        if (fields.empty()) return PyList_New(0);

        std::sort(leaves.begin(), leaves.end());
        arrow::Result<std::shared_ptr<arrow::Table>> table_result;
        {
            ReleaseGIL released;
            table_result = reader->ReadTable(leaves);
        }
        if (python_file && python_file->restore_error()) throw PythonError();
        auto table = unwrap(std::move(table_result));
        arrow::Status validation;
        {
            ReleaseGIL released;
            validation = table->ValidateFull();
        }
        check_status(validation);
        // Parquet returns projected fields in file order, not request order.
        PyRef result(checked(PyList_New(static_cast<Py_ssize_t>(fields.size()))));
        std::vector<int> ordered = fields;
        std::sort(ordered.begin(), ordered.end());
        for (size_t i = 0; i < fields.size(); ++i) {
            auto position = std::lower_bound(ordered.begin(), ordered.end(), fields[i])
                    - ordered.begin();
            PyObject* array = checked(convert_column(*table->column(position)));
            PyList_SET_ITEM(result.get(), static_cast<Py_ssize_t>(i), array);
        }
        return result.release();
    } catch (const PythonError&) {
        if (!PyErr_Occurred()) PyErr_SetString(PyExc_RuntimeError, "Parquet conversion failed");
    } catch (const std::bad_alloc&) {
        PyErr_NoMemory();
    } catch (const std::exception& error) {
        if (!PyErr_Occurred()) PyErr_SetString(PyExc_RuntimeError, error.what());
    } catch (...) {
        if (!PyErr_Occurred()) PyErr_SetString(PyExc_RuntimeError, "unknown native Parquet exception");
    }
    return nullptr;
}

PyMethodDef methods[] = {
    {"parquet_to_arrays", reinterpret_cast<PyCFunction>(read_parquet),
        METH_VARARGS | METH_KEYWORDS,
        "Read Parquet columns into immutable one-dimensional NumPy arrays."},
    {nullptr, nullptr, 0, nullptr}
};

PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "_parquet", nullptr, -1, methods,
    nullptr, nullptr, nullptr, nullptr
};

} // namespace

PyMODINIT_FUNC PyInit__parquet() {
    if (_import_array() < 0) return nullptr;
    return PyModule_Create(&module);
}
