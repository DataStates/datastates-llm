#include "state_provider.hpp"

using namespace datastates;
state_provider_t::state_provider_t(int r, nb::object d_object, std::string key, size_t f_offset, TIER_TYPES device_type)
    : region_id(r), provider_device(device_type), file_start_offset(f_offset), data_key(key) {
    register_state(d_object);
}

state_provider_t::~state_provider_t() {
    delete serializer;
    data_object = nb::none();
}

void state_provider_t::register_state(nb::object d_object) {
    try {
        assert((!d_object.is_none()) && "Object to register cannot be null");
        assert((data_object.is_none()) && "State provider has already registered a data_object state");
        is_tensor = nb::isinstance<nb::ndarray<>>(d_object) || nb::cast<bool>(torch.attr("is_tensor")(d_object));
        is_serialized = nb::isinstance<nb::bytes>(d_object) || nb::isinstance<nb::bytearray>(d_object) || nb::isinstance<nb::str>(d_object);
        // Find the raw size of the data object.
        if (is_tensor) {
            assert(d_object.attr("is_contiguous")() && "Tensor must be contiguous");
            data_size = nb::cast<size_t>(d_object.attr("nbytes"));
            if (nb::cast<bool>(d_object.attr("is_cuda"))) {
                provider_device = GPU_TIER;
            } else if (nb::cast<bool>(d_object.attr("is_pinned")())) {
                provider_device = HOST_PINNED_TIER;
            }
            is_serialized = true; // Tensors are serialized by default
        } else if (is_serialized) {
            data_size = nb::len(d_object);
        } else {
            FATAL("Only tensors or serialized objects (bytes, bytearray, str) are supported currently to register."
                  << " Got object of type: " << nb::cast<std::string>(nb::str(d_object.attr("__class__").attr("__name__")))
                  << " with name " << data_key);
            if (serializer == nullptr) {
                serializer = new pickle_serializer_t(); // Initialize the default serializer
            }
            assert(serializer != nullptr && "Serializer must be initialized for non-serialized objects");
            nb::bytes serialized_data = serializer->serialize(d_object);
            data_size = serialized_data.size();
            is_serialized = false; // Non-tensor objects are not serialized by default
        }
        assert(data_size > 0 && "Data size must be greater than zero");
        data_object = d_object; 
    } catch (std::exception& e) {
        FATAL("Exception caught in register_state: " << e.what());
        data_object = nb::none(); // Reset to none on error
        data_size = 0;
        is_serialized = false;
        is_tensor = false;
        provider_device = HOST_UNPINNED_TIER; // Reset to default tier
        data_status = STATE_PROVIDER_UNREAD_CHUNK; // Reset status
    }
}

TIER_TYPES state_provider_t::get_tier() const {
    return provider_device;
}

size_t state_provider_t::get_data_size() const {
    return data_size;
}

bool state_provider_t::has_next_chunk() const {
    assert(!data_object.is_none() && "Data object is not registered");
    return data_status == STATE_PROVIDER_UNREAD_CHUNK;
}

std::string state_provider_t::get_key() const {
    assert(!data_key.empty() && "Data key is not set");
    return data_key;
}

std::string state_provider_t::get_tensor_shape() const {
    if (is_tensor) {
        nb::object shape = data_object.attr("shape");
        return nb::str(shape).c_str();
    }
    return "";
}

std::string state_provider_t::get_tensor_dtype() const {
    if (is_tensor) {
        try {
            nb::object dtype = data_object.attr("dtype");
            nb::object dtype_str = nb::str(dtype);
            return nb::cast<std::string>(dtype_str);
        } catch (const nb::cast_error& e) {
            FATAL("Failed to get tensor dtype: " << e.what());
        } catch (const std::exception& e) {
            FATAL("Exception caught in get_tensor_dtype: " << e.what());
        }
    }
    return "";
}

void state_provider_t::print_state() const {
    std::cout << "Provider region: " << region_id 
              << ", Tier: " << TIER_TYPE_NAMES[provider_device]
              << ", Data Size: " << data_size 
              << ", Is Tensor: " << (is_tensor ? "Yes" : "No")
              << ", Is Serialized: " << (is_serialized ? "Yes" : "No")
              << ", Data State " << (data_status == STATE_PROVIDER_UNREAD_CHUNK ? "UNREAD" : (data_status == STATE_PROVIDER_CONSUMING_CHUNK ? "CONSUMING" : "CONSUMED"))
              << std::endl;
}

bool state_provider_t::get_next_chunk(TIER_TYPES tier, std::shared_ptr<mem_region_t> dest, size_t chunk_size) {
    try {
        assert(!data_object.is_none() && "Data object is not registered");
        assert(dest != nullptr && "Destination region cannot be null");
        // TODO: Currently the state-provider uses a different memory space when doing serialization.
        // If we want to change this in future, the I/O engine should know and pre-allocate the size of serialized data on the destination region.
        // assert(dest->size == get_data_size() && "Destination region size must match data size: " + std::to_string(dest->size) + " != " + std::to_string(get_data_size()));
        assert(chunk_size == 0 && "Chunking not supported in this provider, chunk_size must be zero");
        dest->uid = region_id;
        dest->file_start_offset = file_start_offset;
        if (tier == provider_device && data_status == STATE_PROVIDER_UNREAD_CHUNK) {
            data_status = STATE_PROVIDER_CONSUMING_CHUNK;
            if (is_tensor) {
                nb::ndarray<> arr = nb::cast<nb::ndarray<>>(data_object);
                dest->ptr = reinterpret_cast<char*>(arr.data());
                dest->size = arr.size() * arr.itemsize();
                DBG("Data object " << dest->uid << " and inner id " << dest->internal_uid << "  is a tensor of shape: " << dest->size);
            } else if (is_serialized) {
                if (nb::isinstance<nb::bytes>(data_object) || nb::isinstance<nb::bytearray>(data_object)) {
                    nb::bytes serialized_data = nb::cast<nb::bytes>(data_object);
                    DBG("Data object " << dest->uid << " and inner id " << dest->internal_uid << "  is a serialized bytes object of size: " << nb::len(data_object) << " with len " << serialized_data.size());
                    dest->ptr = const_cast<char*>(static_cast<const char*>(serialized_data.data()));
                } else {
                    DBG("Data object " << dest->uid << "  is a serialized string of size: " << nb::len(data_object));
                    dest->ptr = const_cast<char*>(nb::cast<std::string>(data_object).c_str());
                }
                dest->size = nb::len(data_object);
            } else {
                assert(serializer != nullptr && "Serializer must be initialized for non-serialized objects");
                DBG("Data object " << dest->uid << " and inner id " << dest->internal_uid << "  is a non serialized string of size: " << nb::len(data_object));
                nb::bytes serialized_data = serializer->serialize(data_object);
                dest->ptr = const_cast<char*>(static_cast<const char*>(serialized_data.data()));
                dest->size = serialized_data.size();
            }
            return true;
        }
        dest->ptr = nullptr; 
        dest->size = 0;
        return false;
    } catch (std::exception& e) {
        FATAL("Exception caught in get_next_chunk: " << e.what() << " for region ID: " << region_id << " and tier: " << TIER_TYPE_NAMES[tier] << " with data size: " << data_size);
        dest->ptr = nullptr; 
        dest->size = 0;
        data_status = STATE_PROVIDER_UNREAD_CHUNK; // Reset status on error
        return false;
    } catch (...) {
        FATAL("Unknown exception caught in get_next_chunk");
        dest->ptr = nullptr; 
        dest->size = 0;
        data_status = STATE_PROVIDER_UNREAD_CHUNK; // Reset status on error
        return false;
    }
}

void state_provider_t::release() {
    assert(data_object.is_none() == false && "Data object must be registered before releasing");
    // assert(data_status == STATE_PROVIDER_CONSUMING_CHUNK && "Region must be in consuming state before releasing");
    data_status = STATE_PROVIDER_UNREAD_CHUNK;
}

