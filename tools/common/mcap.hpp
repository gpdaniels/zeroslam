/*
Copyright (C) 2026 Geoffrey Daniels. https://gpdaniels.com/

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, version 3 of the License only.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

#pragma once
#ifndef ZEROSLAM_TOOLS_COMMON_MCAP_HPP
#define ZEROSLAM_TOOLS_COMMON_MCAP_HPP

#include "file.hpp"
#include "lz4.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

class mcap final {
public:
    struct schema_type {
        unsigned short id = 0;
        std::string name;
        std::string encoding;
        std::vector<unsigned char> data;
    };

    struct channel_type {
        unsigned short id = 0;
        unsigned short schema_id = 0;
        std::string topic;
        std::string message_encoding;
    };

    struct message_type {
        unsigned short channel_id = 0;
        unsigned int sequence = 0;
        unsigned long long log_time = 0;
        unsigned long long publish_time = 0;
        const unsigned char* data = nullptr;
        unsigned long long length = 0;
    };

    struct attachment_type {
        unsigned long long log_time = 0;
        unsigned long long create_time = 0;
        std::string name;
        std::string media_type;
        std::vector<unsigned char> data;
    };

    struct message_index_type {
        unsigned short channel_id = 0;
        unsigned long long log_time = 0;
        unsigned int chunk = 0;
        unsigned long long offset = 0;
    };

    constexpr static const unsigned int no_chunk = ~0u;

private:
    constexpr static const unsigned char magic[8] = { 0x89, 'M', 'C', 'A', 'P', 0x30, '\r', '\n' };

    enum class opcode : unsigned char {
        header = 0x01,
        footer = 0x02,
        schema = 0x03,
        channel = 0x04,
        message = 0x05,
        chunk = 0x06,
        message_index = 0x07,
        chunk_index = 0x08,
        attachment = 0x09,
        attachment_index = 0x0A,
        statistics = 0x0B,
        summary_offset = 0x0E,
        data_end = 0x0F
    };

    struct chunk_type {
        unsigned long long records_offset = 0;
        unsigned long long records_length = 0;
        unsigned long long uncompressed_size = 0;
        unsigned int uncompressed_crc = 0;
        std::string compression;
    };

    constexpr static const size_t chunk_cache_capacity = 4;

    struct cached_chunk_type {
        unsigned int chunk = no_chunk;
        unsigned long long used = 0;
        std::vector<unsigned char> data;
    };

private:
    gtl::file file;
    const unsigned char* memory = nullptr;
    unsigned long long source_length = 0;

    std::string profile_value;
    std::string library_value;
    std::vector<schema_type> schema_records;
    std::vector<channel_type> channel_records;
    std::vector<attachment_type> attachment_records;
    std::vector<chunk_type> chunk_records;
    std::vector<message_index_type> message_index;

    std::vector<cached_chunk_type> chunk_cache;
    unsigned long long cache_clock = 0;
    std::vector<unsigned char> record_buffer;
    std::string read_error;

private:
    static unsigned short read_u16(const unsigned char* data) {
        return static_cast<unsigned short>(static_cast<unsigned int>(data[0]) | (static_cast<unsigned int>(data[1]) << 8));
    }

    static unsigned int read_u32(const unsigned char* data) {
        return static_cast<unsigned int>(data[0]) | (static_cast<unsigned int>(data[1]) << 8) | (static_cast<unsigned int>(data[2]) << 16) | (static_cast<unsigned int>(data[3]) << 24);
    }

    static unsigned long long read_u64(const unsigned char* data) {
        unsigned long long value = 0;
        for (int i = 7; i >= 0; --i) {
            value = (value << 8) | data[i];
        }
        return value;
    }

    static bool read_string(const unsigned char* data, const unsigned long long length, unsigned long long& index, std::string& value) {
        if (length - index < 4) {
            return false;
        }
        const unsigned int string_length = read_u32(&data[index]);
        index += 4;
        if (string_length > length - index) {
            return false;
        }
        value.assign(reinterpret_cast<const char*>(&data[index]), string_length);
        index += string_length;
        return true;
    }

public:
    // The crc-32/iso-hdlc checksum used throughout the mcap format.
    static unsigned int crc32(const unsigned char* data, const unsigned long long length) {
        static unsigned int table[256] = {};
        if (table[1] == 0) {
            for (unsigned int i = 0; i < 256; ++i) {
                unsigned int value = i;
                for (int bit = 0; bit < 8; ++bit) {
                    value = (value & 1) ? (0xEDB88320u ^ (value >> 1)) : (value >> 1);
                }
                table[i] = value;
            }
        }
        unsigned int crc = 0xFFFFFFFFu;
        for (unsigned long long i = 0; i < length; ++i) {
            crc = table[(crc ^ data[i]) & 0xFF] ^ (crc >> 8);
        }
        return crc ^ 0xFFFFFFFFu;
    }

private:
    bool read_source(const unsigned long long offset, const unsigned long long length, unsigned char* const target) const {
        if ((offset > this->source_length) || (length > this->source_length - offset)) {
            return false;
        }
        if (this->memory != nullptr) {
            std::memcpy(target, this->memory + offset, static_cast<size_t>(length));
            return true;
        }
        if (!this->file.set_cursor_position(static_cast<gtl::file::offset_type>(offset))) {
            return false;
        }
        unsigned long long done = 0;
        while (done < length) {
            gtl::file::size_type step = static_cast<gtl::file::size_type>(length - done);
            if (!this->file.read(reinterpret_cast<char*>(target + done), step) || (step == 0)) {
                return false;
            }
            done += step;
        }
        return true;
    }

    bool read_source(const unsigned long long offset, const unsigned long long length, std::vector<unsigned char>& target) const {
        if ((offset > this->source_length) || (length > this->source_length - offset)) {
            return false;
        }
        target.resize(static_cast<size_t>(length));
        return read_source(offset, length, target.data());
    }

    bool parse_chunk_record(const unsigned long long body_offset, const unsigned long long body_length, chunk_type& chunk, std::string& error) const {
        if (body_length < 8 + 8 + 8 + 4 + 4 + 8) {
            error = "invalid chunk record";
            return false;
        }
        unsigned char head[8 + 8 + 8 + 4 + 4];
        if (!read_source(body_offset, sizeof(head), &head[0])) {
            error = "truncated chunk record";
            return false;
        }
        chunk.uncompressed_size = read_u64(&head[16]);
        chunk.uncompressed_crc = read_u32(&head[24]);
        const unsigned int compression_length = read_u32(&head[28]);
        if (static_cast<unsigned long long>(compression_length) + 8 > body_length - 32) {
            error = "invalid chunk record";
            return false;
        }
        std::vector<unsigned char> tail;
        if (!read_source(body_offset + 32, static_cast<unsigned long long>(compression_length) + 8, tail)) {
            error = "truncated chunk record";
            return false;
        }
        chunk.compression.assign(reinterpret_cast<const char*>(tail.data()), compression_length);
        chunk.records_length = read_u64(&tail[compression_length]);
        chunk.records_offset = body_offset + 32 + compression_length + 8;
        if (chunk.records_length > body_length - (32 + compression_length + 8)) {
            error = "invalid chunk record";
            return false;
        }
        if (chunk.uncompressed_size > (1ull << 33)) {
            error = "chunk too large";
            return false;
        }
        if (chunk.compression == "zstd") {
            error = "zstd compressed chunks are unsupported, recompress with none or lz4";
            return false;
        }
        if (!chunk.compression.empty() && (chunk.compression != "lz4")) {
            error = "unknown chunk compression '" + chunk.compression + "'";
            return false;
        }
        return true;
    }

    bool expand_chunk(const chunk_type& chunk, std::vector<unsigned char>& output, std::string& error) const {
        if (chunk.compression.empty()) {
            if (!read_source(chunk.records_offset, chunk.records_length, output)) {
                error = "truncated chunk";
                return false;
            }
        }
        else {
            std::vector<unsigned char> compressed;
            if (!read_source(chunk.records_offset, chunk.records_length, compressed)) {
                error = "truncated chunk";
                return false;
            }
            output.clear();
            output.reserve(static_cast<size_t>(chunk.uncompressed_size));
            if (!lz4::decompress_frame(compressed.data(), compressed.size(), output, error)) {
                return false;
            }
        }
        if (output.size() != chunk.uncompressed_size) {
            error = "chunk decompressed to an unexpected size";
            return false;
        }
        if ((chunk.uncompressed_crc != 0) && (crc32(output.data(), output.size()) != chunk.uncompressed_crc)) {
            error = "chunk crc mismatch";
            return false;
        }
        return true;
    }

    const std::vector<unsigned char>* cached_chunk(const unsigned int index) {
        ++this->cache_clock;
        for (cached_chunk_type& cached : this->chunk_cache) {
            if (cached.chunk == index) {
                cached.used = this->cache_clock;
                return &cached.data;
            }
        }
        if (this->chunk_cache.size() < mcap::chunk_cache_capacity) {
            this->chunk_cache.emplace_back();
        }
        cached_chunk_type* slot = &this->chunk_cache.front();
        for (cached_chunk_type& cached : this->chunk_cache) {
            if (cached.used < slot->used) {
                slot = &cached;
            }
        }
        slot->chunk = no_chunk;
        if (!expand_chunk(this->chunk_records[index], slot->data, this->read_error)) {
            return nullptr;
        }
        slot->chunk = index;
        slot->used = this->cache_clock;
        return &slot->data;
    }

    bool parse_schema_body(const unsigned char* body, const unsigned long long length, std::string& error) {
        schema_type schema;
        if (length < 2) {
            error = "invalid schema record";
            return false;
        }
        schema.id = read_u16(&body[0]);
        unsigned long long index = 2;
        if (!read_string(body, length, index, schema.name) || !read_string(body, length, index, schema.encoding) || (length - index < 4)) {
            error = "invalid schema record";
            return false;
        }
        const unsigned int data_length = read_u32(&body[index]);
        index += 4;
        if (data_length > length - index) {
            error = "invalid schema record";
            return false;
        }
        schema.data.assign(&body[index], &body[index] + data_length);
        for (const schema_type& known : this->schema_records) {
            if (known.id == schema.id) {
                return true;
            }
        }
        this->schema_records.push_back(static_cast<schema_type&&>(schema));
        return true;
    }

    bool parse_channel_body(const unsigned char* body, const unsigned long long length, std::string& error) {
        channel_type channel;
        if (length < 4) {
            error = "invalid channel record";
            return false;
        }
        channel.id = read_u16(&body[0]);
        channel.schema_id = read_u16(&body[2]);
        unsigned long long index = 4;
        if (!read_string(body, length, index, channel.topic) || !read_string(body, length, index, channel.message_encoding)) {
            error = "invalid channel record";
            return false;
        }
        for (const channel_type& known : this->channel_records) {
            if (known.id == channel.id) {
                return true;
            }
        }
        this->channel_records.push_back(static_cast<channel_type&&>(channel));
        return true;
    }

    bool parse_attachment_body(const unsigned char* body, const unsigned long long length, std::string& error) {
        if (length < 16) {
            error = "invalid attachment record";
            return false;
        }
        attachment_type attachment;
        attachment.log_time = read_u64(&body[0]);
        attachment.create_time = read_u64(&body[8]);
        unsigned long long index = 16;
        if (!read_string(body, length, index, attachment.name) || !read_string(body, length, index, attachment.media_type) || (length - index < 8)) {
            error = "invalid attachment record";
            return false;
        }
        const unsigned long long data_length = read_u64(&body[index]);
        index += 8;
        if (data_length > length - index) {
            error = "invalid attachment record";
            return false;
        }
        attachment.data.assign(&body[index], &body[index] + data_length);
        index += data_length;
        if (length - index >= 4) {
            const unsigned int attachment_crc = read_u32(&body[index]);
            if ((attachment_crc != 0) && (crc32(body, index) != attachment_crc)) {
                error = "attachment crc mismatch";
                return false;
            }
        }
        this->attachment_records.push_back(static_cast<attachment_type&&>(attachment));
        return true;
    }

    static bool parse_message_head(const unsigned char* body, const unsigned long long length, message_type& message, std::string& error) {
        if (length < 22) {
            error = "invalid message record";
            return false;
        }
        message.channel_id = read_u16(&body[0]);
        message.sequence = read_u32(&body[2]);
        message.log_time = read_u64(&body[6]);
        message.publish_time = read_u64(&body[14]);
        message.data = &body[22];
        message.length = length - 22;
        return true;
    }

    bool scan_chunk_records(const std::vector<unsigned char>& records, const unsigned int chunk, std::string& error) {
        unsigned long long index = 0;
        while (index < records.size()) {
            if (records.size() - index < 9) {
                error = "truncated record";
                return false;
            }
            const unsigned char record_opcode = records[index];
            const unsigned long long record_length = read_u64(&records[index + 1]);
            if (record_length > records.size() - index - 9) {
                error = "truncated record";
                return false;
            }
            const unsigned char* body = &records[index + 9];
            switch (static_cast<opcode>(record_opcode)) {
                case opcode::schema: {
                    if (!parse_schema_body(body, record_length, error)) {
                        return false;
                    }
                } break;
                case opcode::channel: {
                    if (!parse_channel_body(body, record_length, error)) {
                        return false;
                    }
                } break;
                case opcode::message: {
                    message_type message;
                    if (!parse_message_head(body, record_length, message, error)) {
                        return false;
                    }
                    this->message_index.push_back({ message.channel_id, message.log_time, chunk, index });
                } break;
                case opcode::chunk: {
                    error = "unexpected chunk inside a chunk";
                    return false;
                } break;
                case opcode::header:
                case opcode::footer:
                case opcode::message_index:
                case opcode::chunk_index:
                case opcode::attachment:
                case opcode::attachment_index:
                case opcode::statistics:
                case opcode::summary_offset:
                case opcode::data_end: {
                } break;
            }
            index += 9 + record_length;
        }
        return true;
    }

    bool scan_data_section(std::string& error) {
        unsigned long long index = 8;
        std::vector<unsigned char> body;
        std::vector<unsigned char> records;
        while (index + 8 < this->source_length) {
            unsigned char head[9];
            if (!read_source(index, 9, &head[0])) {
                error = "truncated record";
                return false;
            }
            const unsigned char record_opcode = head[0];
            const unsigned long long record_length = read_u64(&head[1]);
            const unsigned long long body_offset = index + 9;
            if (record_length > this->source_length - body_offset) {
                error = "truncated record";
                return false;
            }
            switch (static_cast<opcode>(record_opcode)) {
                case opcode::header: {
                    if (!read_source(body_offset, record_length, body)) {
                        error = "truncated record";
                        return false;
                    }
                    unsigned long long body_index = 0;
                    if (!read_string(body.data(), record_length, body_index, this->profile_value) || !read_string(body.data(), record_length, body_index, this->library_value)) {
                        error = "invalid header record";
                        return false;
                    }
                } break;
                case opcode::schema: {
                    if (!read_source(body_offset, record_length, body) || !parse_schema_body(body.data(), record_length, error)) {
                        error = error.empty() ? "truncated record" : error;
                        return false;
                    }
                } break;
                case opcode::channel: {
                    if (!read_source(body_offset, record_length, body) || !parse_channel_body(body.data(), record_length, error)) {
                        error = error.empty() ? "truncated record" : error;
                        return false;
                    }
                } break;
                case opcode::attachment: {
                    if (!read_source(body_offset, record_length, body) || !parse_attachment_body(body.data(), record_length, error)) {
                        error = error.empty() ? "truncated record" : error;
                        return false;
                    }
                } break;
                case opcode::message: {
                    unsigned char message_head[22];
                    message_type message;
                    if ((record_length < 22) || !read_source(body_offset, 22, &message_head[0]) || !parse_message_head(&message_head[0], 22, message, error)) {
                        error = "invalid message record";
                        return false;
                    }
                    this->message_index.push_back({ message.channel_id, message.log_time, no_chunk, index });
                } break;
                case opcode::chunk: {
                    chunk_type chunk;
                    if (!parse_chunk_record(body_offset, record_length, chunk, error)) {
                        return false;
                    }
                    if (!expand_chunk(chunk, records, error)) {
                        return false;
                    }
                    this->chunk_records.push_back(chunk);
                    if (!scan_chunk_records(records, static_cast<unsigned int>(this->chunk_records.size() - 1), error)) {
                        return false;
                    }
                } break;
                case opcode::data_end:
                case opcode::footer: {
                    return true;
                } break;
                case opcode::message_index:
                case opcode::chunk_index:
                case opcode::attachment_index:
                case opcode::statistics:
                case opcode::summary_offset: {
                } break;
            }
            index = body_offset + record_length;
        }
        error = "missing data end record";
        return false;
    }

    bool open_from_summary(std::string& error) {
        constexpr unsigned long long footer_length = 1 + 8 + 8 + 8 + 4;
        if (this->source_length < 8 + footer_length + 8) {
            return false;
        }
        unsigned char footer[footer_length];
        if (!read_source(this->source_length - 8 - footer_length, footer_length, &footer[0])) {
            return false;
        }
        if ((static_cast<opcode>(footer[0]) != opcode::footer) || (read_u64(&footer[1]) != 8 + 8 + 4)) {
            return false;
        }
        const unsigned long long summary_start = read_u64(&footer[9]);
        if ((summary_start == 0) || (summary_start >= this->source_length - 8 - footer_length)) {
            return false;
        }
        std::vector<unsigned char> summary;
        if (!read_source(summary_start, this->source_length - 8 - footer_length - summary_start, summary)) {
            return false;
        }
        const unsigned int summary_crc = read_u32(&footer[25]);
        if (summary_crc != 0) {
            std::vector<unsigned char> covered(summary);
            covered.insert(covered.end(), &footer[0], &footer[25]);
            if (crc32(covered.data(), covered.size()) != summary_crc) {
                error = "summary crc mismatch";
                return false;
            }
        }

        struct chunk_index_entry {
            unsigned long long chunk_offset = 0;
            unsigned long long chunk_length = 0;
            std::vector<std::pair<unsigned short, unsigned long long>> message_index_offsets;
        };

        std::vector<chunk_index_entry> chunk_indexes;
        std::vector<unsigned long long> attachment_offsets;
        bool statistics_present = false;
        unsigned long long statistics_message_count = 0;
        unsigned long long index = 0;
        while (index < summary.size()) {
            if (summary.size() - index < 9) {
                error = "truncated summary record";
                return false;
            }
            const unsigned char record_opcode = summary[index];
            const unsigned long long record_length = read_u64(&summary[index + 1]);
            if (record_length > summary.size() - index - 9) {
                error = "truncated summary record";
                return false;
            }
            const unsigned char* body = &summary[index + 9];
            switch (static_cast<opcode>(record_opcode)) {
                case opcode::schema: {
                    if (!parse_schema_body(body, record_length, error)) {
                        return false;
                    }
                } break;
                case opcode::channel: {
                    if (!parse_channel_body(body, record_length, error)) {
                        return false;
                    }
                } break;
                case opcode::chunk_index: {
                    if (record_length < 8 + 8 + 8 + 8 + 4) {
                        error = "invalid chunk index record";
                        return false;
                    }
                    chunk_index_entry entry;
                    entry.chunk_offset = read_u64(&body[16]);
                    entry.chunk_length = read_u64(&body[24]);
                    const unsigned int offsets_length = read_u32(&body[32]);
                    if ((offsets_length % 10 != 0) || (offsets_length > record_length - 36)) {
                        error = "invalid chunk index record";
                        return false;
                    }
                    for (unsigned int offset = 0; offset < offsets_length; offset += 10) {
                        entry.message_index_offsets.push_back({ read_u16(&body[36 + offset]), read_u64(&body[38 + offset]) });
                    }
                    chunk_indexes.push_back(static_cast<chunk_index_entry&&>(entry));
                } break;
                case opcode::attachment_index: {
                    if (record_length < 8) {
                        error = "invalid attachment index record";
                        return false;
                    }
                    attachment_offsets.push_back(read_u64(&body[0]));
                } break;
                case opcode::statistics: {
                    if (record_length < 8) {
                        error = "invalid statistics record";
                        return false;
                    }
                    statistics_present = true;
                    statistics_message_count = read_u64(&body[0]);
                } break;
                case opcode::header:
                case opcode::footer:
                case opcode::message:
                case opcode::chunk:
                case opcode::message_index:
                case opcode::attachment:
                case opcode::summary_offset:
                case opcode::data_end: {
                } break;
            }
            index += 9 + record_length;
        }
        if (chunk_indexes.empty()) {
            return false;
        }
        {
            unsigned char head[9];
            if (read_source(8, 9, &head[0]) && (static_cast<opcode>(head[0]) == opcode::header)) {
                std::vector<unsigned char> body;
                unsigned long long body_index = 0;
                if (read_source(17, read_u64(&head[1]), body)) {
                    static_cast<void>(read_string(body.data(), body.size(), body_index, this->profile_value) && read_string(body.data(), body.size(), body_index, this->library_value));
                }
            }
        }
        std::vector<unsigned char> body;
        std::vector<unsigned char> records;
        for (const chunk_index_entry& entry : chunk_indexes) {
            unsigned char head[9];
            if (!read_source(entry.chunk_offset, 9, &head[0]) || (static_cast<opcode>(head[0]) != opcode::chunk)) {
                error = "chunk index points at no chunk";
                return false;
            }
            chunk_type chunk;
            if (!parse_chunk_record(entry.chunk_offset + 9, read_u64(&head[1]), chunk, error)) {
                return false;
            }
            this->chunk_records.push_back(chunk);
            const unsigned int chunk_number = static_cast<unsigned int>(this->chunk_records.size() - 1);
            const size_t first_entry = this->message_index.size();
            bool indexed = !entry.message_index_offsets.empty();
            for (const std::pair<unsigned short, unsigned long long>& message_index_offset : entry.message_index_offsets) {
                if (!read_source(message_index_offset.second, 9, &head[0]) || (static_cast<opcode>(head[0]) != opcode::message_index)) {
                    indexed = false;
                    break;
                }
                const unsigned long long record_length = read_u64(&head[1]);
                if ((record_length < 6) || !read_source(message_index_offset.second + 9, record_length, body)) {
                    indexed = false;
                    break;
                }
                const unsigned short channel_id = read_u16(&body[0]);
                const unsigned int entries_length = read_u32(&body[2]);
                if ((entries_length % 16 != 0) || (entries_length > record_length - 6)) {
                    indexed = false;
                    break;
                }
                for (unsigned int offset = 0; offset < entries_length; offset += 16) {
                    this->message_index.push_back({ channel_id, read_u64(&body[6 + offset]), chunk_number, read_u64(&body[14 + offset]) });
                }
            }
            if (!indexed) {
                this->message_index.resize(first_entry);
                if (!expand_chunk(chunk, records, error) || !scan_chunk_records(records, chunk_number, error)) {
                    return false;
                }
            }
            else {
                std::sort(this->message_index.begin() + static_cast<std::ptrdiff_t>(first_entry), this->message_index.end(), [](const message_index_type& lhs, const message_index_type& rhs) {
                    return lhs.offset < rhs.offset;
                });
            }
        }
        if (statistics_present && (statistics_message_count != this->message_index.size())) {
            return false;
        }
        for (const unsigned long long offset : attachment_offsets) {
            unsigned char head[9];
            if (!read_source(offset, 9, &head[0]) || (static_cast<opcode>(head[0]) != opcode::attachment)) {
                error = "attachment index points at no attachment";
                return false;
            }
            if (!read_source(offset + 9, read_u64(&head[1]), body) || !parse_attachment_body(body.data(), body.size(), error)) {
                error = error.empty() ? "truncated attachment" : error;
                return false;
            }
        }
        return true;
    }

    bool open_source(std::string& error) {
        error.clear();
        unsigned char head[8];
        unsigned char tail[8];
        if ((this->source_length < 8 + 8) || !read_source(0, 8, &head[0]) || !read_source(this->source_length - 8, 8, &tail[0])) {
            error = "not an mcap file";
            return false;
        }
        for (size_t i = 0; i < 8; ++i) {
            if ((head[i] != magic[i]) || (tail[i] != magic[i])) {
                error = "not an mcap file";
                return false;
            }
        }
        if (!open_from_summary(error)) {
            error.clear();
            this->schema_records.clear();
            this->channel_records.clear();
            this->attachment_records.clear();
            this->chunk_records.clear();
            this->message_index.clear();
            if (!scan_data_section(error)) {
                return false;
            }
        }
        std::stable_sort(this->message_index.begin(), this->message_index.end(), [](const message_index_type& lhs, const message_index_type& rhs) {
            return lhs.log_time < rhs.log_time;
        });
        return true;
    }

public:
    mcap() = default;
    mcap(const mcap&) = delete;
    mcap& operator=(const mcap&) = delete;
    mcap(mcap&&) = delete;
    mcap& operator=(mcap&&) = delete;
    ~mcap() = default;

    bool open(const std::string& path, std::string& error) {
        this->close();
        if (!this->file.open(path.c_str(), gtl::file::access_type::read_only, gtl::file::creation_type::open_only, gtl::file::cursor_type::start_of_file)) {
            error = "the file could not be opened";
            return false;
        }
        gtl::file::size_type size = 0;
        if (!this->file.get_size(size)) {
            error = "the file size could not be read";
            this->close();
            return false;
        }
        this->source_length = size;
        if (!open_source(error)) {
            this->close();
            return false;
        }
        return true;
    }

    bool parse(const unsigned char* data, const unsigned long long length, std::string& error) {
        this->close();
        if (data == nullptr) {
            error = "not an mcap file";
            return false;
        }
        this->memory = data;
        this->source_length = length;
        if (!open_source(error)) {
            this->close();
            return false;
        }
        return true;
    }

    void close() {
        this->file.close();
        this->memory = nullptr;
        this->source_length = 0;
        this->profile_value.clear();
        this->library_value.clear();
        this->schema_records.clear();
        this->channel_records.clear();
        this->attachment_records.clear();
        this->chunk_records.clear();
        this->message_index.clear();
        this->chunk_cache.clear();
        this->cache_clock = 0;
        this->record_buffer.clear();
        this->read_error.clear();
    }

    bool is_open() const {
        return (this->memory != nullptr) || this->file.is_open();
    }

    const std::string& get_profile() const {
        return profile_value;
    }

    const std::string& get_library() const {
        return library_value;
    }

    const std::vector<schema_type>& get_schemas() const {
        return schema_records;
    }

    const std::vector<channel_type>& get_channels() const {
        return channel_records;
    }

    const std::vector<attachment_type>& get_attachments() const {
        return attachment_records;
    }

    const std::vector<message_index_type>& get_message_index() const {
        return message_index;
    }

    size_t get_message_count() const {
        return message_index.size();
    }

    size_t get_chunk_count() const {
        return chunk_records.size();
    }

    bool read_message(const size_t index, message_type& message) {
        if (index >= this->message_index.size()) {
            this->read_error = "message index out of range";
            return false;
        }
        const message_index_type& entry = this->message_index[index];
        const unsigned char* records = nullptr;
        unsigned long long available = 0;
        if (entry.chunk == no_chunk) {
            unsigned char head[9];
            if (!read_source(entry.offset, 9, &head[0]) || (static_cast<opcode>(head[0]) != opcode::message)) {
                this->read_error = "message index points at no message";
                return false;
            }
            const unsigned long long record_length = read_u64(&head[1]);
            if (!read_source(entry.offset + 9, record_length, this->record_buffer)) {
                this->read_error = "truncated message record";
                return false;
            }
            return parse_message_head(this->record_buffer.data(), record_length, message, this->read_error);
        }
        const std::vector<unsigned char>* const chunk = cached_chunk(entry.chunk);
        if (chunk == nullptr) {
            return false;
        }
        records = chunk->data();
        available = chunk->size();
        if ((entry.offset > available) || (available - entry.offset < 9) || (static_cast<opcode>(records[entry.offset]) != opcode::message)) {
            this->read_error = "message index points at no message";
            return false;
        }
        const unsigned long long record_length = read_u64(&records[entry.offset + 1]);
        if (record_length > available - entry.offset - 9) {
            this->read_error = "truncated message record";
            return false;
        }
        return parse_message_head(&records[entry.offset + 9], record_length, message, this->read_error);
    }

    const std::string& get_read_error() const {
        return read_error;
    }

    const schema_type* find_schema(const unsigned short id) const {
        for (const schema_type& schema : schema_records) {
            if (schema.id == id) {
                return &schema;
            }
        }
        return nullptr;
    }

    const channel_type* find_channel(const std::string& topic) const {
        for (const channel_type& channel : channel_records) {
            if (channel.topic == topic) {
                return &channel;
            }
        }
        return nullptr;
    }

public:
    class writer final {
    private:
        struct chunk_index_type {
            unsigned long long start_time = 0;
            unsigned long long end_time = 0;
            unsigned long long chunk_offset = 0;
            unsigned long long chunk_length = 0;
            std::vector<std::pair<unsigned short, unsigned long long>> message_index_offsets;
            unsigned long long message_index_length = 0;
            unsigned long long compressed_size = 0;
            unsigned long long uncompressed_size = 0;
        };

        struct chunk_message_type {
            unsigned short channel_id = 0;
            unsigned long long log_time = 0;
            unsigned long long offset = 0;
        };

        struct stream_type {
            std::vector<unsigned char> pending;
            std::vector<chunk_message_type> messages;
            unsigned long long start_time = 0;
            unsigned long long end_time = 0;
            bool has_messages = false;
        };

        constexpr static const unsigned long long bulk_threshold = 16ull << 10;
        constexpr static const unsigned long long chunk_size = 8ull << 20;

    private:
        gtl::file file;
        bool failed = false;
        unsigned long long written = 0;
        std::string chunk_compression;
        stream_type bulk;
        stream_type compact;
        unsigned short next_schema_id = 1;
        unsigned short next_channel_id = 0;
        // The summary section state.
        std::vector<std::vector<unsigned char>> schema_bodies;
        std::vector<std::vector<unsigned char>> channel_bodies;
        std::vector<chunk_index_type> chunk_indexes;
        std::vector<std::pair<unsigned short, unsigned long long>> channel_message_counts;
        unsigned long long message_count = 0;
        unsigned long long message_start_time = 0;
        unsigned long long message_end_time = 0;

    private:
        static void put_u16(std::vector<unsigned char>& buffer, const unsigned short value) {
            buffer.push_back(static_cast<unsigned char>(value & 0xFF));
            buffer.push_back(static_cast<unsigned char>((value >> 8) & 0xFF));
        }

        static void put_u32(std::vector<unsigned char>& buffer, const unsigned int value) {
            for (int i = 0; i < 4; ++i) {
                buffer.push_back(static_cast<unsigned char>((value >> (8 * i)) & 0xFF));
            }
        }

        static void put_u64(std::vector<unsigned char>& buffer, const unsigned long long value) {
            for (int i = 0; i < 8; ++i) {
                buffer.push_back(static_cast<unsigned char>((value >> (8 * i)) & 0xFF));
            }
        }

        static void put_string(std::vector<unsigned char>& buffer, const std::string& value) {
            put_u32(buffer, static_cast<unsigned int>(value.size()));
            buffer.insert(buffer.end(), value.begin(), value.end());
        }

        static void put_record(std::vector<unsigned char>& target, const opcode record_opcode, const std::vector<unsigned char>& body) {
            target.push_back(static_cast<unsigned char>(record_opcode));
            put_u64(target, body.size());
            target.insert(target.end(), body.begin(), body.end());
        }

        void emit(const std::vector<unsigned char>& bytes) {
            if (this->failed) {
                return;
            }
            size_t index = 0;
            while (index < bytes.size()) {
                gtl::file::size_type step = bytes.size() - index;
                if (!this->file.write(reinterpret_cast<const char*>(bytes.data() + index), step) || (step == 0)) {
                    this->failed = true;
                    return;
                }
                index += step;
            }
            this->written += bytes.size();
        }

        void emit_record(const opcode record_opcode, const std::vector<unsigned char>& body) {
            std::vector<unsigned char> record;
            put_record(record, record_opcode, body);
            emit(record);
        }

        void flush_chunk(stream_type& stream) {
            if (stream.pending.empty()) {
                return;
            }
            chunk_index_type index;
            index.start_time = stream.has_messages ? stream.start_time : 0;
            index.end_time = stream.has_messages ? stream.end_time : 0;
            index.chunk_offset = this->written;
            index.uncompressed_size = stream.pending.size();
            std::vector<unsigned char> body;
            put_u64(body, index.start_time);
            put_u64(body, index.end_time);
            put_u64(body, stream.pending.size());
            put_u32(body, crc32(stream.pending.data(), stream.pending.size()));
            put_string(body, this->chunk_compression);
            const std::vector<unsigned char> compressed = lz4::compress_frame(stream.pending.data(), stream.pending.size());
            index.compressed_size = compressed.size();
            put_u64(body, compressed.size());
            body.insert(body.end(), compressed.begin(), compressed.end());
            emit_record(opcode::chunk, body);
            index.chunk_length = this->written - index.chunk_offset;
            // One message index record per channel of the chunk, in first seen order.
            const unsigned long long message_indexes_start = this->written;
            std::vector<unsigned short> channels;
            for (const chunk_message_type& message : stream.messages) {
                bool known = false;
                for (const unsigned short channel : channels) {
                    known = known || (channel == message.channel_id);
                }
                if (!known) {
                    channels.push_back(message.channel_id);
                }
            }
            for (const unsigned short channel : channels) {
                index.message_index_offsets.push_back({ channel, this->written });
                std::vector<unsigned char> index_body;
                put_u16(index_body, channel);
                unsigned int entries = 0;
                for (const chunk_message_type& message : stream.messages) {
                    entries += (message.channel_id == channel);
                }
                put_u32(index_body, entries * 16);
                for (const chunk_message_type& message : stream.messages) {
                    if (message.channel_id == channel) {
                        put_u64(index_body, message.log_time);
                        put_u64(index_body, message.offset);
                    }
                }
                emit_record(opcode::message_index, index_body);
            }
            index.message_index_length = this->written - message_indexes_start;
            this->chunk_indexes.push_back(static_cast<chunk_index_type&&>(index));
            stream.messages.clear();
            stream.pending.clear();
            stream.has_messages = false;
        }

    public:
        writer() = default;
        writer(const writer&) = delete;
        writer& operator=(const writer&) = delete;
        writer(writer&&) = delete;
        writer& operator=(writer&&) = delete;
        ~writer() = default;

        bool begin(const std::string& path, const std::string& profile, const std::string& library, const std::string& compression = "") {
            this->file.close();
            this->failed = false;
            this->written = 0;
            this->chunk_compression = compression;
            this->bulk = stream_type();
            this->compact = stream_type();
            this->next_schema_id = 1;
            this->next_channel_id = 0;
            this->schema_bodies.clear();
            this->channel_bodies.clear();
            this->chunk_indexes.clear();
            this->channel_message_counts.clear();
            this->message_count = 0;
            this->message_start_time = 0;
            this->message_end_time = 0;
            if (!this->file.open(path.c_str(), gtl::file::access_type::write_only, gtl::file::creation_type::create_or_open, gtl::file::cursor_type::start_of_truncated)) {
                this->failed = true;
                return false;
            }
            emit(std::vector<unsigned char>(&magic[0], &magic[0] + 8));
            std::vector<unsigned char> body;
            put_string(body, profile);
            put_string(body, library);
            emit_record(opcode::header, body);
            return !this->failed;
        }

        bool has_failed() const {
            return this->failed;
        }

        unsigned long long get_written() const {
            return this->written;
        }

        unsigned short add_schema(const std::string& name, const std::string& encoding, const std::string& data) {
            const unsigned short id = this->next_schema_id++;
            std::vector<unsigned char> body;
            put_u16(body, id);
            put_string(body, name);
            put_string(body, encoding);
            put_u32(body, static_cast<unsigned int>(data.size()));
            body.insert(body.end(), data.begin(), data.end());
            emit_record(opcode::schema, body);
            this->schema_bodies.push_back(body);
            return id;
        }

        unsigned short add_channel(const unsigned short schema_id, const std::string& topic, const std::string& message_encoding) {
            const unsigned short id = this->next_channel_id++;
            std::vector<unsigned char> body;
            put_u16(body, id);
            put_u16(body, schema_id);
            put_string(body, topic);
            put_string(body, message_encoding);
            put_u32(body, 0); // An empty metadata map.
            emit_record(opcode::channel, body);
            this->channel_bodies.push_back(body);
            return id;
        }

        void add_message(const unsigned short channel_id, const unsigned int sequence, const unsigned long long log_time, const unsigned long long publish_time, const unsigned char* data, const unsigned long long length) {
            std::vector<unsigned char> body;
            body.reserve(static_cast<std::size_t>(22 + length));
            put_u16(body, channel_id);
            put_u32(body, sequence);
            put_u64(body, log_time);
            put_u64(body, publish_time);
            body.insert(body.end(), data, data + length);
            if ((this->message_count == 0) || (log_time < this->message_start_time)) {
                this->message_start_time = log_time;
            }
            if ((this->message_count == 0) || (log_time > this->message_end_time)) {
                this->message_end_time = log_time;
            }
            ++this->message_count;
            bool counted = false;
            for (std::pair<unsigned short, unsigned long long>& count : this->channel_message_counts) {
                if (count.first == channel_id) {
                    ++count.second;
                    counted = true;
                }
            }
            if (!counted) {
                this->channel_message_counts.push_back({ channel_id, 1 });
            }
            if (this->chunk_compression.empty()) {
                emit_record(opcode::message, body);
                return;
            }
            stream_type& stream = (length >= writer::bulk_threshold) ? this->bulk : this->compact;
            if (!stream.has_messages || (log_time < stream.start_time)) {
                stream.start_time = log_time;
            }
            if (!stream.has_messages || (log_time > stream.end_time)) {
                stream.end_time = log_time;
            }
            stream.has_messages = true;
            stream.messages.push_back({ channel_id, log_time, stream.pending.size() });
            put_record(stream.pending, opcode::message, body);
            if (stream.pending.size() > writer::chunk_size) {
                flush_chunk(stream);
            }
        }

        bool finish() {
            flush_chunk(this->compact);
            flush_chunk(this->bulk);
            std::vector<unsigned char> body;
            put_u32(body, 0); // No data section crc.
            emit_record(opcode::data_end, body);
            body.clear();
            if (this->chunk_compression.empty()) {
                // The unchunked form keeps an empty summary section.
                put_u64(body, 0);
                put_u64(body, 0);
                put_u32(body, 0);
                emit_record(opcode::footer, body);
                emit(std::vector<unsigned char>(&magic[0], &magic[0] + 8));
                this->file.close();
                return !this->failed;
            }
            // The summary section: schemas, channels, chunk indexes, and statistics, each group located by a summary offset record.
            std::vector<unsigned char> summary;
            const unsigned long long summary_start = this->written;

            struct group_type {
                opcode group_opcode;
                unsigned long long start;
                unsigned long long length;
            };

            std::vector<group_type> groups;
            const auto begin_group = [&](const opcode group_opcode) {
                groups.push_back({ group_opcode, summary_start + summary.size(), 0 });
            };
            const auto end_group = [&]() {
                groups.back().length = summary_start + summary.size() - groups.back().start;
            };
            if (!this->schema_bodies.empty()) {
                begin_group(opcode::schema);
                for (const std::vector<unsigned char>& schema : this->schema_bodies) {
                    put_record(summary, opcode::schema, schema);
                }
                end_group();
            }
            if (!this->channel_bodies.empty()) {
                begin_group(opcode::channel);
                for (const std::vector<unsigned char>& channel : this->channel_bodies) {
                    put_record(summary, opcode::channel, channel);
                }
                end_group();
            }
            if (!this->chunk_indexes.empty()) {
                begin_group(opcode::chunk_index);
                for (const chunk_index_type& index : this->chunk_indexes) {
                    std::vector<unsigned char> index_body;
                    put_u64(index_body, index.start_time);
                    put_u64(index_body, index.end_time);
                    put_u64(index_body, index.chunk_offset);
                    put_u64(index_body, index.chunk_length);
                    put_u32(index_body, static_cast<unsigned int>(index.message_index_offsets.size() * 10));
                    for (const std::pair<unsigned short, unsigned long long>& offset : index.message_index_offsets) {
                        put_u16(index_body, offset.first);
                        put_u64(index_body, offset.second);
                    }
                    put_u64(index_body, index.message_index_length);
                    put_string(index_body, this->chunk_compression);
                    put_u64(index_body, index.compressed_size);
                    put_u64(index_body, index.uncompressed_size);
                    put_record(summary, opcode::chunk_index, index_body);
                }
                end_group();
            }
            {
                begin_group(opcode::statistics);
                std::vector<unsigned char> statistics_body;
                put_u64(statistics_body, this->message_count);
                put_u16(statistics_body, static_cast<unsigned short>(this->schema_bodies.size()));
                put_u32(statistics_body, static_cast<unsigned int>(this->channel_bodies.size()));
                put_u32(statistics_body, 0); // No attachment records.
                put_u32(statistics_body, 0); // No metadata records.
                put_u32(statistics_body, static_cast<unsigned int>(this->chunk_indexes.size()));
                put_u64(statistics_body, this->message_start_time);
                put_u64(statistics_body, this->message_end_time);
                put_u32(statistics_body, static_cast<unsigned int>(this->channel_message_counts.size() * 10));
                for (const std::pair<unsigned short, unsigned long long>& count : this->channel_message_counts) {
                    put_u16(statistics_body, count.first);
                    put_u64(statistics_body, count.second);
                }
                put_record(summary, opcode::statistics, statistics_body);
                end_group();
            }
            const unsigned long long summary_offset_start = summary_start + summary.size();
            for (const group_type& group : groups) {
                std::vector<unsigned char> offset_body;
                offset_body.push_back(static_cast<unsigned char>(group.group_opcode));
                put_u64(offset_body, group.start);
                put_u64(offset_body, group.length);
                put_record(summary, opcode::summary_offset, offset_body);
            }
            // The footer, with the summary crc covering the summary section and the footer
            // fields that precede the crc itself.
            summary.push_back(static_cast<unsigned char>(opcode::footer));
            put_u64(summary, 8 + 8 + 4);
            put_u64(summary, summary_start);
            put_u64(summary, summary_offset_start);
            const unsigned int summary_crc = crc32(summary.data(), summary.size());
            put_u32(summary, summary_crc);
            summary.insert(summary.end(), &magic[0], &magic[0] + 8);
            emit(summary);
            this->file.close();
            return !this->failed;
        }
    };
};

#endif // ZEROSLAM_TOOLS_COMMON_MCAP_HPP
