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

#include "lz4.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <string>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

// A reader and a minimal writer for the mcap container format (https://mcap.dev/spec),
// the ROS2 recording format. The reader scans the data section (the summary section is
// not needed), handles chunked and unchunked files, and supports uncompressed and lz4
// compressed chunks; zstd compressed chunks are rejected. The writer produces valid
// unchunked files with an empty summary section.
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
        const unsigned char* data = nullptr;
        unsigned long long length = 0;
    };

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

private:
    std::string profile_value;
    std::string library_value;
    std::vector<schema_type> schema_records;
    std::vector<channel_type> channel_records;
    std::vector<message_type> message_records;
    std::vector<attachment_type> attachment_records;
    // Owns the decompressed data of lz4 chunks, referenced by the message records.
    std::vector<std::vector<unsigned char>> chunk_buffers;

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
    // Parse the schema, channel, and message records of the data section or of one
    // decompressed chunk.
    bool parse_records(const unsigned char* data, const unsigned long long length, const bool inside_chunk, bool& reached_end, std::string& error) {
        unsigned long long index = 0;
        while (index < length) {
            if (length - index < 9) {
                error = "truncated record";
                return false;
            }
            const unsigned char record_opcode = data[index];
            const unsigned long long record_length = read_u64(&data[index + 1]);
            index += 9;
            if (record_length > length - index) {
                error = "truncated record";
                return false;
            }
            const unsigned char* body = &data[index];
            unsigned long long body_index = 0;
            switch (static_cast<opcode>(record_opcode)) {
                case opcode::header: {
                    if (!read_string(body, record_length, body_index, profile_value) || !read_string(body, record_length, body_index, library_value)) {
                        error = "invalid header record";
                        return false;
                    }
                } break;
                case opcode::schema: {
                    schema_type schema;
                    if (record_length < 2) {
                        error = "invalid schema record";
                        return false;
                    }
                    schema.id = read_u16(&body[0]);
                    body_index = 2;
                    if (!read_string(body, record_length, body_index, schema.name) || !read_string(body, record_length, body_index, schema.encoding)) {
                        error = "invalid schema record";
                        return false;
                    }
                    if ((record_length - body_index < 4)) {
                        error = "invalid schema record";
                        return false;
                    }
                    const unsigned int data_length = read_u32(&body[body_index]);
                    body_index += 4;
                    if (data_length > record_length - body_index) {
                        error = "invalid schema record";
                        return false;
                    }
                    schema.data.assign(&body[body_index], &body[body_index] + data_length);
                    schema_records.push_back(static_cast<schema_type&&>(schema));
                } break;
                case opcode::channel: {
                    channel_type channel;
                    if (record_length < 4) {
                        error = "invalid channel record";
                        return false;
                    }
                    channel.id = read_u16(&body[0]);
                    channel.schema_id = read_u16(&body[2]);
                    body_index = 4;
                    if (!read_string(body, record_length, body_index, channel.topic) || !read_string(body, record_length, body_index, channel.message_encoding)) {
                        error = "invalid channel record";
                        return false;
                    }
                    channel_records.push_back(static_cast<channel_type&&>(channel));
                } break;
                case opcode::message: {
                    if (record_length < 22) {
                        error = "invalid message record";
                        return false;
                    }
                    message_type message;
                    message.channel_id = read_u16(&body[0]);
                    message.sequence = read_u32(&body[2]);
                    message.log_time = read_u64(&body[6]);
                    message.publish_time = read_u64(&body[14]);
                    message.data = &body[22];
                    message.length = record_length - 22;
                    message_records.push_back(message);
                } break;
                case opcode::chunk: {
                    if (inside_chunk) {
                        error = "unexpected chunk inside a chunk";
                        return false;
                    }
                    if (record_length < 8 + 8 + 8 + 4) {
                        error = "invalid chunk record";
                        return false;
                    }
                    const unsigned long long uncompressed_size = read_u64(&body[16]);
                    const unsigned int uncompressed_crc = read_u32(&body[24]);
                    body_index = 28;
                    std::string compression;
                    if (!read_string(body, record_length, body_index, compression)) {
                        error = "invalid chunk record";
                        return false;
                    }
                    if (record_length - body_index < 8) {
                        error = "invalid chunk record";
                        return false;
                    }
                    const unsigned long long records_length = read_u64(&body[body_index]);
                    body_index += 8;
                    if (records_length > record_length - body_index) {
                        error = "invalid chunk record";
                        return false;
                    }
                    const unsigned char* records = &body[body_index];
                    if (uncompressed_size > (1ull << 33)) {
                        error = "chunk too large";
                        return false;
                    }
                    if (compression.empty()) {
                        if ((uncompressed_crc != 0) && (crc32(records, records_length) != uncompressed_crc)) {
                            error = "chunk crc mismatch";
                            return false;
                        }
                        if (!parse_records(records, records_length, true, reached_end, error)) {
                            return false;
                        }
                    }
                    else if (compression == "lz4") {
                        chunk_buffers.emplace_back();
                        std::vector<unsigned char>& buffer = chunk_buffers.back();
                        buffer.reserve(static_cast<std::size_t>(uncompressed_size));
                        if (!lz4::decompress_frame(records, records_length, buffer, error)) {
                            return false;
                        }
                        if (buffer.size() != uncompressed_size) {
                            error = "chunk decompressed to an unexpected size";
                            return false;
                        }
                        if ((uncompressed_crc != 0) && (crc32(buffer.data(), buffer.size()) != uncompressed_crc)) {
                            error = "chunk crc mismatch";
                            return false;
                        }
                        if (!parse_records(buffer.data(), buffer.size(), true, reached_end, error)) {
                            return false;
                        }
                    }
                    else if (compression == "zstd") {
                        error = "zstd compressed chunks are unsupported, recompress with none or lz4";
                        return false;
                    }
                    else {
                        error = "unknown chunk compression '" + compression + "'";
                        return false;
                    }
                } break;
                case opcode::attachment: {
                    if (record_length < 16) {
                        error = "invalid attachment record";
                        return false;
                    }
                    attachment_type attachment;
                    attachment.log_time = read_u64(&body[0]);
                    attachment.create_time = read_u64(&body[8]);
                    body_index = 16;
                    if (!read_string(body, record_length, body_index, attachment.name) || !read_string(body, record_length, body_index, attachment.media_type)) {
                        error = "invalid attachment record";
                        return false;
                    }
                    if (record_length - body_index < 8) {
                        error = "invalid attachment record";
                        return false;
                    }
                    const unsigned long long data_length = read_u64(&body[body_index]);
                    body_index += 8;
                    if (data_length > record_length - body_index) {
                        error = "invalid attachment record";
                        return false;
                    }
                    attachment.data = &body[body_index];
                    attachment.length = data_length;
                    body_index += data_length;
                    if (record_length - body_index >= 4) {
                        const unsigned int attachment_crc = read_u32(&body[body_index]);
                        if ((attachment_crc != 0) && (crc32(body, body_index) != attachment_crc)) {
                            error = "attachment crc mismatch";
                            return false;
                        }
                    }
                    attachment_records.push_back(attachment);
                } break;
                case opcode::data_end:
                case opcode::footer: {
                    // The records after the data section are indexes, not needed.
                    reached_end = true;
                    return true;
                } break;
                default: {
                    // Unknown and index records are skipped.
                } break;
            }
            index += record_length;
        }
        return true;
    }

public:
    // Parse an mcap file from memory. The message and attachment records reference the
    // given buffer (and internal chunk buffers), it must outlive this object.
    bool parse(const unsigned char* data, const unsigned long long length, std::string& error) {
        profile_value.clear();
        library_value.clear();
        schema_records.clear();
        channel_records.clear();
        message_records.clear();
        attachment_records.clear();
        chunk_buffers.clear();
        if ((length < 8 + 8) || (data == nullptr)) {
            error = "too short to be an mcap file";
            return false;
        }
        for (int i = 0; i < 8; ++i) {
            if ((data[i] != magic[i]) || (data[length - 8 + i] != magic[i])) {
                error = "not an mcap file";
                return false;
            }
        }
        bool reached_end = false;
        if (!parse_records(&data[8], length - 16, false, reached_end, error)) {
            return false;
        }
        if (!reached_end) {
            error = "missing data end record";
            return false;
        }
        return true;
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

    const std::vector<message_type>& get_messages() const {
        return message_records;
    }

    const std::vector<attachment_type>& get_attachments() const {
        return attachment_records;
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
    // A minimal mcap writer. With a compression the records are batched into lz4
    // compressed chunks and the file is fully indexed (message indexes, chunk indexes,
    // statistics, and a summary section), which streaming web viewers
    // require; without one the records are written plain and the summary is left empty.
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

    private:
        std::vector<unsigned char> output;
        std::vector<unsigned char> pending;
        std::string chunk_compression;
        unsigned long long chunk_start_time = 0;
        unsigned long long chunk_end_time = 0;
        bool chunk_has_messages = false;
        unsigned short next_schema_id = 1;
        unsigned short next_channel_id = 0;
        // The summary section state.
        std::vector<std::vector<unsigned char>> schema_bodies;
        std::vector<std::vector<unsigned char>> channel_bodies;
        std::vector<chunk_index_type> chunk_indexes;
        std::vector<chunk_message_type> chunk_messages;
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

        // Batched records go into the pending chunk when compressing, otherwise straight out.
        std::vector<unsigned char>& batched() {
            return chunk_compression.empty() ? output : pending;
        }

        void flush_chunk() {
            if (pending.empty()) {
                return;
            }
            chunk_index_type index;
            index.start_time = chunk_has_messages ? chunk_start_time : 0;
            index.end_time = chunk_has_messages ? chunk_end_time : 0;
            index.chunk_offset = output.size();
            index.uncompressed_size = pending.size();
            std::vector<unsigned char> body;
            put_u64(body, index.start_time);
            put_u64(body, index.end_time);
            put_u64(body, pending.size());
            put_u32(body, crc32(pending.data(), pending.size()));
            put_string(body, chunk_compression);
            const std::vector<unsigned char> compressed = lz4::compress_frame(pending.data(), pending.size());
            index.compressed_size = compressed.size();
            put_u64(body, compressed.size());
            body.insert(body.end(), compressed.begin(), compressed.end());
            put_record(output, opcode::chunk, body);
            index.chunk_length = output.size() - index.chunk_offset;
            // One message index record per channel of the chunk, in first seen order.
            const unsigned long long message_indexes_start = output.size();
            std::vector<unsigned short> channels;
            for (const chunk_message_type& message : chunk_messages) {
                bool known = false;
                for (const unsigned short channel : channels) {
                    known = known || (channel == message.channel_id);
                }
                if (!known) {
                    channels.push_back(message.channel_id);
                }
            }
            for (const unsigned short channel : channels) {
                index.message_index_offsets.push_back({ channel, output.size() });
                std::vector<unsigned char> index_body;
                put_u16(index_body, channel);
                unsigned int entries = 0;
                for (const chunk_message_type& message : chunk_messages) {
                    entries += (message.channel_id == channel);
                }
                put_u32(index_body, entries * 16);
                for (const chunk_message_type& message : chunk_messages) {
                    if (message.channel_id == channel) {
                        put_u64(index_body, message.log_time);
                        put_u64(index_body, message.offset);
                    }
                }
                put_record(output, opcode::message_index, index_body);
            }
            index.message_index_length = output.size() - message_indexes_start;
            chunk_indexes.push_back(static_cast<chunk_index_type&&>(index));
            chunk_messages.clear();
            pending.clear();
            chunk_has_messages = false;
        }

    public:
        // An empty compression writes plain unchunked records, "lz4" batches them into
        // compressed chunks.
        void begin(const std::string& profile, const std::string& library, const std::string& compression = "") {
            output.assign(&magic[0], &magic[0] + 8);
            pending.clear();
            chunk_compression = compression;
            chunk_has_messages = false;
            next_schema_id = 1;
            next_channel_id = 0;
            schema_bodies.clear();
            channel_bodies.clear();
            chunk_indexes.clear();
            chunk_messages.clear();
            channel_message_counts.clear();
            message_count = 0;
            message_start_time = 0;
            message_end_time = 0;
            std::vector<unsigned char> body;
            put_string(body, profile);
            put_string(body, library);
            put_record(output, opcode::header, body);
        }

        unsigned short add_schema(const std::string& name, const std::string& encoding, const std::string& data) {
            const unsigned short id = next_schema_id++;
            std::vector<unsigned char> body;
            put_u16(body, id);
            put_string(body, name);
            put_string(body, encoding);
            put_u32(body, static_cast<unsigned int>(data.size()));
            body.insert(body.end(), data.begin(), data.end());
            put_record(batched(), opcode::schema, body);
            schema_bodies.push_back(body);
            return id;
        }

        unsigned short add_channel(const unsigned short schema_id, const std::string& topic, const std::string& message_encoding) {
            const unsigned short id = next_channel_id++;
            std::vector<unsigned char> body;
            put_u16(body, id);
            put_u16(body, schema_id);
            put_string(body, topic);
            put_string(body, message_encoding);
            put_u32(body, 0); // An empty metadata map.
            put_record(batched(), opcode::channel, body);
            channel_bodies.push_back(body);
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
            if (!chunk_has_messages || (log_time < chunk_start_time)) {
                chunk_start_time = log_time;
            }
            if (!chunk_has_messages || (log_time > chunk_end_time)) {
                chunk_end_time = log_time;
            }
            chunk_has_messages = true;
            if ((message_count == 0) || (log_time < message_start_time)) {
                message_start_time = log_time;
            }
            if ((message_count == 0) || (log_time > message_end_time)) {
                message_end_time = log_time;
            }
            ++message_count;
            bool counted = false;
            for (std::pair<unsigned short, unsigned long long>& count : channel_message_counts) {
                if (count.first == channel_id) {
                    ++count.second;
                    counted = true;
                }
            }
            if (!counted) {
                channel_message_counts.push_back({ channel_id, 1 });
            }
            chunk_messages.push_back({ channel_id, log_time, pending.size() });
            put_record(batched(), opcode::message, body);
            if (!chunk_compression.empty() && (pending.size() > (8ull << 20))) {
                flush_chunk();
            }
        }

        const std::vector<unsigned char>& finish() {
            flush_chunk();
            std::vector<unsigned char> body;
            put_u32(body, 0); // No data section crc.
            put_record(output, opcode::data_end, body);
            body.clear();
            if (chunk_compression.empty()) {
                // The unchunked form keeps an empty summary section.
                put_u64(body, 0);
                put_u64(body, 0);
                put_u32(body, 0);
                put_record(output, opcode::footer, body);
                output.insert(output.end(), &magic[0], &magic[0] + 8);
                return output;
            }
            // The summary section: schemas, channels, chunk indexes, and statistics, each group located by a summary offset record.
            const unsigned long long summary_start = output.size();

            struct group_type {
                opcode group_opcode;
                unsigned long long start;
                unsigned long long length;
            };

            std::vector<group_type> groups;
            const auto begin_group = [&](const opcode group_opcode) {
                groups.push_back({ group_opcode, output.size(), 0 });
            };
            const auto end_group = [&]() {
                groups.back().length = output.size() - groups.back().start;
            };
            if (!schema_bodies.empty()) {
                begin_group(opcode::schema);
                for (const std::vector<unsigned char>& schema : schema_bodies) {
                    put_record(output, opcode::schema, schema);
                }
                end_group();
            }
            if (!channel_bodies.empty()) {
                begin_group(opcode::channel);
                for (const std::vector<unsigned char>& channel : channel_bodies) {
                    put_record(output, opcode::channel, channel);
                }
                end_group();
            }
            if (!chunk_indexes.empty()) {
                begin_group(opcode::chunk_index);
                for (const chunk_index_type& index : chunk_indexes) {
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
                    put_string(index_body, chunk_compression);
                    put_u64(index_body, index.compressed_size);
                    put_u64(index_body, index.uncompressed_size);
                    put_record(output, opcode::chunk_index, index_body);
                }
                end_group();
            }
            {
                begin_group(opcode::statistics);
                std::vector<unsigned char> statistics_body;
                put_u64(statistics_body, message_count);
                put_u16(statistics_body, static_cast<unsigned short>(schema_bodies.size()));
                put_u32(statistics_body, static_cast<unsigned int>(channel_bodies.size()));
                put_u32(statistics_body, 0); // No attachment records.
                put_u32(statistics_body, 0); // No metadata records.
                put_u32(statistics_body, static_cast<unsigned int>(chunk_indexes.size()));
                put_u64(statistics_body, message_start_time);
                put_u64(statistics_body, message_end_time);
                put_u32(statistics_body, static_cast<unsigned int>(channel_message_counts.size() * 10));
                for (const std::pair<unsigned short, unsigned long long>& count : channel_message_counts) {
                    put_u16(statistics_body, count.first);
                    put_u64(statistics_body, count.second);
                }
                put_record(output, opcode::statistics, statistics_body);
                end_group();
            }
            const unsigned long long summary_offset_start = output.size();
            for (const group_type& group : groups) {
                std::vector<unsigned char> offset_body;
                offset_body.push_back(static_cast<unsigned char>(group.group_opcode));
                put_u64(offset_body, group.start);
                put_u64(offset_body, group.length);
                put_record(output, opcode::summary_offset, offset_body);
            }
            // The footer, with the summary crc covering the summary section and the footer
            // fields that precede the crc itself.
            output.push_back(static_cast<unsigned char>(opcode::footer));
            put_u64(output, 8 + 8 + 4);
            put_u64(output, summary_start);
            put_u64(output, summary_offset_start);
            const unsigned int summary_crc = crc32(&output[summary_start], output.size() - summary_start);
            put_u32(output, summary_crc);
            output.insert(output.end(), &magic[0], &magic[0] + 8);
            return output;
        }
    };
};

#endif // ZEROSLAM_TOOLS_COMMON_MCAP_HPP
