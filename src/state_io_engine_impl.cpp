#include "state_io_engine_impl.hpp"


namespace datastates {

state_io_engine_impl_t::state_io_engine_impl_t(size_t host_cache_size, int gpu_id, int rank_, bool use_io_uring, size_t fs_block_alignment): 
    rank(rank_), use_io_uring(use_io_uring), fs_block_alignment(fs_block_alignment) {
    try {
        core_engine = create_core_engine(host_cache_size, gpu_id, rank_, use_io_uring, fs_block_alignment);
    } catch(std::exception& e) {
        FATAL("Standard exception caught in datastates init: " << e.what());
    }
}

void state_io_engine_impl_t::ckpt(std::uint64_t version, state_manager_t* state, std::string path) {
    try {
        for (TIER_TYPES tier : {GPU_TIER, HOST_UNPINNED_TIER, HOST_PINNED_TIER}) {
            while (state->has_next_chunk(tier)) {
                std::shared_ptr<mem_region_t> m = std::make_shared<mem_region_t>(version, 0 /*region_id*/, nullptr /*ptr*/, 0 /*size*/ , 0 /*file_offset*/, path, tier);
                state->get_next_chunk(tier, m);
                DBG("Going to checkpoint memory region with UID " << m->uid << " of size " << m->size << " at file offset " << m->file_start_offset);
                core_engine->ckpt_region(m);
            }
        }
        // Write at the top of the file, where to find the header.
        size_t header_begin_offset = state->get_file_offset();
        char* ptr = reinterpret_cast<char*>(&header_begin_offset);
        std::shared_ptr<mem_region_t> m_header_offset = std::make_shared<mem_region_t>(version, state->get_state_provider_uid(), ptr, sizeof(size_t), 0, path, HOST_UNPINNED_TIER);
        core_engine->ckpt_region(m_header_offset);

        // Write the header to the file.
        std::string header = state->get_state_meta().data();
        size_t header_size = header.size();
        char* header_ptr = header.data();
        std::shared_ptr<mem_region_t> m_header = std::make_shared<mem_region_t>(version, state->get_state_provider_uid(), header_ptr, header_size, header_begin_offset, path, HOST_UNPINNED_TIER);
        core_engine->ckpt_region(m_header);
        
    } catch (std::exception &e) {
        FATAL("Exception caught in ckpt." << e.what());
    }
    return;
}

void state_io_engine_impl_t::wait(state_manager_t* state, bool persist) {
    try {
        core_engine->wait(persist);
        state->release();
    }  catch (std::exception &e) {
        FATAL("Exception caught in wait D2H." << e.what());
    }
}

std::string state_io_engine_impl_t::get_queue_stats(bool for_flush_queue) {
    try {
        return core_engine->get_queue_stats(for_flush_queue);
    } catch (std::exception &e) {
        FATAL("Exception caught in get_queue_stats." << e.what());
    }
}

std::string state_io_engine_impl_t::restore(std::uint64_t version, std::string path) {
    try {
        DBG("[state_io_engine_impl] Starting restore for path: " << path << " version: " << version);

        // 1) Read header begin offset stored at file offset 0 (size_t)
        size_t header_begin_offset = 0;
        {
            // buffer for reading the begin offset
            size_t tmp = 0;
            char *tmp_ptr = reinterpret_cast<char*>(&tmp);
            auto m_header_offset = std::make_shared<mem_region_t>(version, 0 /*region_id*/, tmp_ptr, sizeof(size_t), 0 /*file_offset*/, path, HOST_UNPINNED_TIER);
            core_engine->restore_region(m_header_offset);
            header_begin_offset = tmp;
        }

        // Sanity
        if (header_begin_offset == 0) {
            FATAL("[state_io_engine_impl] header_begin_offset is 0; cannot restore header for: " << path);
        }

        // 2) Read the header from file (header is from header_begin_offset to EOF)
        size_t file_size = std::filesystem::file_size(path);
        if (file_size <= header_begin_offset) {
            FATAL("[state_io_engine_impl] file size (" << file_size << ") is <= header offset (" 
                << header_begin_offset << ") for: " << path);
        }
        size_t header_size = file_size - header_begin_offset;

        // allocate a temporary buffer for header (we'll construct std::string then strip null padding)
        std::string header_buf(header_size, '\0');
        {
            auto m_header = std::make_shared<mem_region_t>(version, 0 /*region_id*/, header_buf.data(), header_size, header_begin_offset, path, HOST_UNPINNED_TIER);
            core_engine->restore_region(m_header);
        }

        // Strip trailing null padding (get_state_meta pads with '\0')
        auto null_pos = header_buf.find('\0');
        std::string header_json_str;
        if (null_pos != std::string::npos) {
            header_json_str = header_buf.substr(0, null_pos);
        } else {
            header_json_str = header_buf;
        }

        if (header_json_str.empty()) {
            FATAL("[state_io_engine_impl] header JSON is empty after stripping padding for: " << path);
        }

        // Parse header JSON
        nlohmann::json header_json;
        try {
            header_json = nlohmann::json::parse(header_json_str);
        } catch (const std::exception &e) {
            FATAL("[state_io_engine_impl] Failed to parse header JSON: " << e.what() << " raw header (len=" << header_json_str.size() << ")");
        }

        // The header_json is state_meta: iterate keys and allocate+restore region for each
        // produce an 'out_json' mapping key -> { "ptr": <uintptr>, "size": <size>, "dtype":..., "shape":..., "offsets":[s,e] }
        nlohmann::json out_json = nlohmann::json::object();

        for (auto header_it = header_json.begin(); header_it != header_json.end(); ++header_it) {
            const std::string &key = header_it.key();
            const auto &meta = header_it.value();

            // Offsets are mandatory
            if (!meta.contains("offsets") || !meta["offsets"].is_array() || meta["offsets"].size() != 2) {
                FATAL("[state_io_engine_impl] Skipping key (no valid offsets): " << key << " meta: " << meta.dump());
            }
            size_t start = meta["offsets"][0].get<size_t>();
            size_t end   = meta["offsets"][1].get<size_t>();
            if (end < start) {
                FATAL("[state_io_engine_impl] Skipping key (end < start): " << key);
            }
            size_t data_size = end - start;

            // Allocate aligned host buffer (aligned to fs_block_alignment)
            void* buf = nullptr;
            int rc = posix_memalign(&buf, fs_block_alignment, data_size);
            if (rc != 0 || buf == nullptr) {
                FATAL("[state_io_engine_impl] posix_memalign failed for key " << key << " rc=" << rc);
            }
            std::memset(buf, 0, data_size);

            // Keep allocation alive
            persistent_allocs.push_back(buf);

            // Create mem_region_t that points to the allocated buffer and ask core_engine to restore into it.
            auto m = std::make_shared<mem_region_t>(version,
                                                   0 /*region_id*/,
                                                   reinterpret_cast<char*>(buf),
                                                   data_size,
                                                   start /*file_offset*/,
                                                   path,
                                                   HOST_UNPINNED_TIER);

            DBG("[state_io_engine_impl] Restoring key=" << key << " size=" << data_size << " offset=" << start << " -> buf=" << buf);
            core_engine->restore_region(m);

            // Build metadata for Python
            nlohmann::json entry = nlohmann::json::object();
            // pointer as integer (uintptr_t) so Python can use ctypes if desired
            entry["ptr"] = reinterpret_cast<std::uintptr_t>(buf);
            entry["size"] = data_size;
            entry["offsets"] = meta["offsets"];

            if (meta.contains("dtype")) {
                entry["dtype"] = meta["dtype"];
            }
            if (meta.contains("shape")) {
                entry["shape"] = meta["shape"];
            }

            out_json[key] = std::move(entry);
        }
        return out_json.dump();
    } catch (std::exception &e) {
        FATAL("Exception caught in restore: " << e.what());
    }
}



std::string state_io_engine_impl_t::shutdown() {
    try {
        if (!is_state_io_engine_active) {
            return "State I/O engine already shutdown.";
        }
        std::string res = core_engine->shutdown();
        is_state_io_engine_active = false;
        persistent_allocs.clear();
        return res;
    } catch (std::exception &e) {
        FATAL("Exception caught in shutdown." << e.what());
    }
}

state_io_engine_impl_t::~state_io_engine_impl_t() {
    try {
        shutdown();
    } catch (std::exception &e) {
        FATAL("Exception caught in destructor." << e.what());
    }
}

} // namespace datastates