#ifndef __VLCC_STATES_COMPOSITE_PROVIDER_HPP
#define __VLCC_STATES_COMPOSITE_PROVIDER_HPP

#include <vector>
#include <memory>
#include <map>
#include "state_provider.hpp"
#include "rawptr_state_provider.hpp"
#include "object_state_provider.hpp"


namespace vlcc_states {
class composite_state_provider_t : public state_provider_t {
private:
    std::vector<std::shared_ptr<state_provider_t>> providers; // List of registered state providers
    std::map<TIER_TYPES, int> current_provider_index;
public:
    composite_state_provider_t(std::string name, TIER_TYPES device_type=COMPOSITE_TIER)
        : state_provider_t(std::move(name), device_type) {}

    ~composite_state_provider_t() override {
        providers.clear();
        current_provider_index.clear();
    }

    void register_provider(std::shared_ptr<state_provider_t> provider) {
        assert(provider != nullptr && "Provider cannot be null");
        providers.push_back(provider);
        if (current_provider_index.find(provider->get_tier()) == current_provider_index.end()) {
            current_provider_index[provider->get_tier()] = 0; // Initialize index for this tier
        }
    }

    std::shared_ptr<state_region_t> get_next_chunk(TIER_TYPES tier) override {
        assert(!providers.empty() && "No providers registered");

        int& start = current_provider_index[tier];
        for (int i = start; i < providers.size(); ++i) {
            if (providers[i]->get_tier() != tier)
                continue;
            auto chunk = providers[i]->get_next_chunk(tier);
            if (chunk) {
                start = i + 1;
                return chunk;
            }
        }
        current_provider_index[tier] = providers.size();
        return nullptr;
    }

    void release() override {
        assert(!providers.empty() && "No providers registered to release");
        for (auto& e: current_provider_index) {
            assert(e.second == providers.size() && "All providers should be consumed before releasing, found unconsumed on tier: " + std::to_string(e.first));
            e.second = 0; // Reset index for each tier
        }
        for (auto& provider : providers) {
            provider->release();
        }
    }
}; // class composite_state_provider_t

} // namespace vlcc_states

#endif // __VLCC_STATES_COMPOSITE_PROVIDER_HPP