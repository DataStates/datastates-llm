#include <state_provider.hpp>
#include "rawptr_state_provider.hpp"
#include "object_state_provider.hpp"
#include "composite_state_provider.hpp"

using namespace vlcc_states;

std::string state_provider_t::get_name() {
    return provider_name;
}

TIER_TYPES state_provider_t::get_tier() {
    return provider_device;
}

std::shared_ptr<state_provider_t> create_rawptr_provider(const std::string& name, TIER_TYPES tier, void* ptr, size_t size) {
    return std::make_shared<rawptr_state_provider_t>(name, tier, ptr, size);
}

std::shared_ptr<state_provider_t> create_object_provider(const std::string& name, TIER_TYPES tier, nanobind::object nb_object) {
    return std::make_shared<object_state_provider_t>(name, tier, nb_object);
}

std::shared_ptr<state_provider_t> create_composite_provider(const std::string& name, TIER_TYPES tier) {
    return std::make_shared<composite_state_provider_t>(name, tier);
}


