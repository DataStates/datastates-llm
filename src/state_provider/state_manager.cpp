
#include "state_manager.hpp"
using namespace datastates;
state_manager_t::state_manager_t() {
    // The first size_t bytes are reserved for the header start offset.
    file_offset = std::max(get_fs_block_alignment(), sizeof(size_t));
}

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

int state_manager_t::get_state_provider_uid() {
    return state_provider_uid++;
}

void state_manager_t::add_var(nb::object data, std::string key) {
    try {
        int id = get_state_provider_uid();
        assert(!key.empty() && "Key cannot be empty");
        assert(!data.is_none() && "Data to register cannot be null");
        assert(ids.find(id) == ids.end() && "ID already registered");
        ids.insert(id);
        auto provider = std::make_shared<state_provider_t>(id, data, key, file_offset);
        register_provider(provider);
        compute_meta(provider);
        DBG("[DataStates][Add_var] Registered new state provider with ID: " << id 
                  << ", size: " << provider->get_data_size() 
                  << ", tier: " << TIER_TYPE_NAMES[provider->get_tier()] 
                  << ", relative file offset: " << provider->file_start_offset);
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

void state_manager_t::compute_meta(std::shared_ptr<state_provider_t> provider) {
    try {
        assert(state_meta.find(provider->get_key()) == state_meta.end() && "Key already exists in meta");
        size_t data_size = provider->get_data_size();
        if (provider->is_tensor) {
            state_meta[provider->get_key()] = {
                {"shape", provider->get_tensor_shape()},
                {"dtype", provider->get_tensor_dtype()},
                {"offsets", {file_offset, file_offset + data_size}},
            };
        } else if (provider->is_serialized) {
            state_meta[provider->get_key()] = {
                {"offsets", {file_offset, file_offset + data_size}},
            };
        } else {
            assert(false && "Unserialized data is not supported yet for computing headers");
        }
        file_offset = get_aligned_offset(file_offset + data_size);
    } catch (std::exception& e) {
        FATAL("Exception caught in compute_meta: " << e.what());
    }
}

void state_manager_t::print_state() {
    try {
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
        it->second = -1; // Mark as fully consumed
        return false;
    } catch (std::exception& e) {
        FATAL("Exception caught in has_next_chunk: " << e.what());
        return false;
    }
}

bool state_manager_t::get_next_chunk(TIER_TYPES tier, std::shared_ptr<mem_region_t> dest, size_t chunk_size) {
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
        current_provider_index[tier] = -1; // Mark as fully consumed
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
            // if (e.second != -1) {
            //     FATAL("Not all providers are consumed for tier " << e.first << " name " << TIER_TYPE_NAMES[e.first] << ". Consumed: " << e.second << ", Total: " << providers.size());
            // }
            e.second = 0; // Reset index for each tier
        }
        for (auto& provider : providers) {
            provider->release();
        }
    } catch (std::exception& e) {
        FATAL("Exception caught in release: " << e.what());
    }
}


size_t state_manager_t::get_file_offset() const {
    return file_offset;
}

std::string state_manager_t::get_state_meta() const {
    try {
        std::string res = state_meta.dump();
        if (res.size() % get_fs_block_alignment() != 0) {
            res += std::string(get_fs_block_alignment() - (res.size() % get_fs_block_alignment()), '\0');
        }
        return res;
    } catch (std::exception& e) {
        FATAL("Exception caught in get_state_meta: " << e.what());
    }
}