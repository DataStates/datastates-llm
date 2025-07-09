
#include "state_manager.hpp"
using namespace datastates;
state_manager_t::state_manager_t() {}

state_manager_t::~state_manager_t() {
    try {
        for (auto& provider : providers) {
            provider->release();
        }
        providers.clear();
        current_provider_index.clear();
    } catch (std::exception& e) {
        FATAL("Exception caught in state_manager_t destructor: " << e.what());
    }
}

void state_manager_t::add_var(nb::object data) {
    try {
        int id = state_provider_uid++;
        assert(!data.is_none() && "Data to register cannot be null");
        assert(ids.find(id) == ids.end() && "ID already registered");
        ids.insert(id);
        auto provider = std::make_shared<state_provider_t>(id, data, relative_file_offset);
        register_provider(provider);
        DBG("[DataStates][Add_var] Registered new state provider with ID: " << id 
                  << ", size: " << provider->get_data_size() 
                  << ", tier: " << TIER_TYPE_NAMES[provider->get_tier()] 
                  << ", relative file offset: " << provider->file_start_offset);
        relative_file_offset += provider->get_data_size();
    } catch (std::exception& e) {
        FATAL("Exception caught in add_var: " << e.what());
    }
}

void state_manager_t::register_provider(std::shared_ptr<state_provider_t> provider) {
    try {
        assert(provider != nullptr && "Provider cannot be null");
        providers.push_back(provider);
        if (current_provider_index.find(provider->get_tier()) == current_provider_index.end()) {
            current_provider_index[provider->get_tier()] = 0; // Initialize index for this tier
        }
    } catch (std::exception& e) {
        FATAL("Exception caught in register_provider: " << e.what());
    }
}

void state_manager_t::print_state() {
    try {
        std::cout << "Number of Registered Providers: " << providers.size() << std::endl;
        for (const auto& provider : providers) {
            provider->print_state();
        }
    } catch (std::exception& e) {
        FATAL("Exception caught in print_state: " << e.what());
    }
}

bool state_manager_t::has_next_chunk(TIER_TYPES tier) {
    try {
        assert(!providers.empty() && "No providers registered");
        auto it = current_provider_index.find(tier);
        if (it == current_provider_index.end()) {
            return false; // No providers for this tier
        }
        int& start = it->second;
        for (int i = start; i < providers.size(); ++i) {
            if (providers[i]->get_tier() == tier && providers[i]->has_next_chunk()) {
                return true;
            }
        }
        return false;
    } catch (std::exception& e) {
        FATAL("Exception caught in has_next_chunk: " << e.what());
        return false;
    }
}

bool state_manager_t::get_next_chunk(TIER_TYPES tier, mem_region_t* dest, size_t chunk_size) {
    try {
        assert(!providers.empty() && "No providers registered");
        int& start = current_provider_index[tier];
        for (int i = start; i < providers.size(); ++i) {
            if (providers[i]->get_tier() != tier)
                continue;
            auto chunk = providers[i]->get_next_chunk(tier, dest, chunk_size);
            if (chunk) {
                start = i + 1;
                return chunk;
            }
        }
        current_provider_index[tier] = providers.size();
        return false;
    } catch (std::exception& e) {
        FATAL("Exception caught in get_next_chunk: " << e.what());
        return false;
    }
}

void state_manager_t::release() {
    try {
        assert(!providers.empty() && "No providers registered to release");
        for (auto& e: current_provider_index) {
            assert(e.second == providers.size() && "All providers should be consumed before releasing, found unconsumed on tier: " + std::to_string(e.first));
            e.second = 0; // Reset index for each tier
        }
        for (auto& provider : providers) {
            provider->release();
        }
    } catch (std::exception& e) {
        FATAL("Exception caught in release: " << e.what());
    }
}
