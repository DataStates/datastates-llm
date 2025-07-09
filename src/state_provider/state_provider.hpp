#ifndef __STATE_PROVIDER_HPP
#define __STATE_PROVIDER_HPP

#include <iostream>
#include <string>
#include <memory>
#define NDEBUG
#include <cassert>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include "serializers/pickle_serializer.hpp"
#include "common/defs.hpp"
#include "common/utils.hpp"
#include "common/mem_region.hpp"

namespace datastates {
static nb::object torch = nb::module_::import_("torch");
class state_provider_t {
protected:
    int region_id = 0; // Unique identifier for the state provider
    TIER_TYPES provider_device = HOST_UNPINNED_TIER; // Default to unregistered tier
    STATE_PROVIDER_CHUNK_STATUS data_status = STATE_PROVIDER_UNREAD_CHUNK;
    size_t data_size = 0;

    pickle_serializer_t* serializer = nullptr;
    nb::object data_object = nb::none(); // The Python object to be serialized
    bool is_serialized = false; // Flag to check if the object is serialized
    bool is_tensor = false; // Flag to check if the object is a tensor
    size_t chunk_size = STATE_PROVIDER_DEFAULT_CHUNK_SIZE; // Default chunk size
    void register_state(nb::object d_object);
public:
    size_t file_start_offset = 0; // Start offset in file for this region
    state_provider_t(int region_id, nb::object d_object, size_t f_offset, TIER_TYPES device_type = HOST_UNPINNED_TIER);
    ~state_provider_t();
    TIER_TYPES get_tier() const; /* E.g. GPU, CPU, UVM, etc. */
    size_t get_data_size() const;
    void print_state() const;
    bool has_next_chunk() const;
    bool get_next_chunk(TIER_TYPES tier, mem_region_t* dest, size_t chunk_size = 0);
    void release();
};

} // namespace datastates

#endif //__STATE_PROVIDER_HPP
