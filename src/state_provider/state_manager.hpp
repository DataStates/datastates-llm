#ifndef __STATE_MANAGER_HPP
#define __STATE_MANAGER_HPP

#include <vector>
#include <memory>
#include <map>
#include <set>
#include "state_provider.hpp"
#include "datastates.hpp"

namespace datastates {
class state_manager_t {
private:
    std::vector<std::shared_ptr<state_provider_t>> providers; // List of registered state providers
    std::map<TIER_TYPES, int> current_provider_index;
    std::set<int> ids; // Set of unique IDs for registered providers
    size_t relative_file_offset = 0;
    int state_provider_uid = 1;
public:
    state_manager_t();
    ~state_manager_t();
    void register_provider(std::shared_ptr<state_provider_t> provider);
    void add_var(nb::object data);
    bool has_next_chunk(TIER_TYPES tier);
    bool get_next_chunk(TIER_TYPES tier, mem_region_t* dest, size_t chunk_size = 0);
    void print_state();
    void release();
}; // class state_manager_t

} // namespace datastates


#endif // __STATE_MANAGER_HPP