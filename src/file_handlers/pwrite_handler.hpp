#ifndef __DATASTATES_FILE_HANDLERS_PWRITE_HANDLER_HPP
#define __DATASTATES_FILE_HANDLERS_PWRITE_HANDLER_HPP

#include "base_file_handler.hpp"
#include <unordered_map>
#include <string>

namespace datastates {
class pwrite_handler_t: public base_file_handler_t {
    public:
        pwrite_handler_t(std::shared_ptr<mem_pool_t> pool);
        ~pwrite_handler_t();
        void write(std::shared_ptr<mem_region_t> m, bool is_odirect=false) override;
        void read(std::shared_ptr<mem_region_t> m, bool is_odirect=false) override;
        void fsync() override;
};
}

#endif // __DATASTATES_FILE_HANDLERS_PWRITE_HANDLER_HPP