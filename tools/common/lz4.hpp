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
#ifndef ZEROSLAM_COMMON_LZ4_HPP
#define ZEROSLAM_COMMON_LZ4_HPP

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <string>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

// A decoder and a greedy encoder for the lz4 frame and block formats, enough to read and
// write lz4 compressed mcap chunks. Dictionaries are not supported and checksums are
// skipped on read, not verified.
class lz4 final {
private:
    constexpr static const unsigned int frame_magic = 0x184D2204u;
    constexpr static const unsigned int skippable_magic_first = 0x184D2A50u;
    constexpr static const unsigned int skippable_magic_last = 0x184D2A5Fu;

private:
    static unsigned int read_u32(const unsigned char* data) {
        return static_cast<unsigned int>(data[0]) | (static_cast<unsigned int>(data[1]) << 8) | (static_cast<unsigned int>(data[2]) << 16) | (static_cast<unsigned int>(data[3]) << 24);
    }

    // The xxhash32 of the frame descriptor, required in the frame header checksum byte.
    static unsigned int xxh32(const unsigned char* data, const unsigned long long length) {
        constexpr static const unsigned int prime_1 = 2654435761u;
        constexpr static const unsigned int prime_2 = 2246822519u;
        constexpr static const unsigned int prime_3 = 3266489917u;
        constexpr static const unsigned int prime_4 = 668265263u;
        constexpr static const unsigned int prime_5 = 374761393u;
        const auto rotate = [](const unsigned int value, const int amount) {
            return (value << amount) | (value >> (32 - amount));
        };
        unsigned long long index = 0;
        unsigned int hash;
        if (length >= 16) {
            unsigned int state[4] = { prime_1 + prime_2, prime_2, 0, static_cast<unsigned int>(0) - prime_1 };
            while (index + 16 <= length) {
                for (int lane = 0; lane < 4; ++lane) {
                    state[lane] = rotate(state[lane] + read_u32(&data[index]) * prime_2, 13) * prime_1;
                    index += 4;
                }
            }
            hash = rotate(state[0], 1) + rotate(state[1], 7) + rotate(state[2], 12) + rotate(state[3], 18);
        }
        else {
            hash = prime_5;
        }
        hash += static_cast<unsigned int>(length);
        while (index + 4 <= length) {
            hash = rotate(hash + read_u32(&data[index]) * prime_3, 17) * prime_4;
            index += 4;
        }
        while (index < length) {
            hash = rotate(hash + data[index] * prime_5, 11) * prime_1;
            ++index;
        }
        hash ^= hash >> 15;
        hash *= prime_2;
        hash ^= hash >> 13;
        hash *= prime_3;
        hash ^= hash >> 16;
        return hash;
    }

public:
    // Decompress one lz4 block, appending to the output. Matches may reference bytes
    // already in the output, which supports both independent and dependent frame blocks.
    static bool decompress_block(const unsigned char* source, const unsigned long long source_length, std::vector<unsigned char>& output, std::string& error) {
        unsigned long long index = 0;
        while (true) {
            if (index >= source_length) {
                error = "truncated lz4 block";
                return false;
            }
            const unsigned char token = source[index++];
            unsigned long long literal_length = token >> 4;
            if (literal_length == 15) {
                while (true) {
                    if (index >= source_length) {
                        error = "truncated lz4 literal length";
                        return false;
                    }
                    const unsigned char extra = source[index++];
                    literal_length += extra;
                    if (extra != 255) {
                        break;
                    }
                }
            }
            if (literal_length > source_length - index) {
                error = "truncated lz4 literals";
                return false;
            }
            output.insert(output.end(), &source[index], &source[index] + literal_length);
            index += literal_length;
            if (index == source_length) {
                // The final sequence of a block is literals only.
                return true;
            }
            if (source_length - index < 2) {
                error = "truncated lz4 match offset";
                return false;
            }
            const unsigned long long offset = static_cast<unsigned long long>(source[index]) | (static_cast<unsigned long long>(source[index + 1]) << 8);
            index += 2;
            if ((offset == 0) || (offset > output.size())) {
                error = "invalid lz4 match offset";
                return false;
            }
            unsigned long long match_length = (token & 0x0F) + 4;
            if ((token & 0x0F) == 15) {
                while (true) {
                    if (index >= source_length) {
                        error = "truncated lz4 match length";
                        return false;
                    }
                    const unsigned char extra = source[index++];
                    match_length += extra;
                    if (extra != 255) {
                        break;
                    }
                }
            }
            // A byte at a time as the match may overlap the bytes it produces.
            std::size_t position = output.size() - static_cast<std::size_t>(offset);
            for (unsigned long long i = 0; i < match_length; ++i) {
                output.push_back(output[position + i]);
            }
        }
    }

    // Compress one lz4 block with a greedy four byte hash table matcher.
    static void compress_block(const unsigned char* source, const unsigned long long source_length, std::vector<unsigned char>& output) {
        const auto emit_length = [&output](unsigned long long value) {
            while (value >= 255) {
                output.push_back(255);
                value -= 255;
            }
            output.push_back(static_cast<unsigned char>(value));
        };
        unsigned long long anchor = 0;
        if (source_length >= 13) {
            constexpr static const unsigned int table_bits = 16;
            std::vector<unsigned int> table(1u << table_bits, 0xFFFFFFFFu);
            const unsigned long long match_limit = source_length - 12;
            const unsigned long long end_limit = source_length - 5;
            unsigned long long index = 0;
            while (index < match_limit) {
                const unsigned int key = (read_u32(&source[index]) * 2654435761u) >> (32 - table_bits);
                const unsigned int candidate = table[key];
                table[key] = static_cast<unsigned int>(index);
                if ((candidate != 0xFFFFFFFFu) && (index - candidate <= 65535) && (read_u32(&source[candidate]) == read_u32(&source[index]))) {
                    unsigned long long match_end = index + 4;
                    unsigned long long reference = candidate + 4;
                    while ((match_end < end_limit) && (source[match_end] == source[reference])) {
                        ++match_end;
                        ++reference;
                    }
                    const unsigned long long literal_length = index - anchor;
                    const unsigned long long match_length = match_end - index - 4;
                    output.push_back(static_cast<unsigned char>(((literal_length < 15 ? literal_length : 15) << 4) | (match_length < 15 ? match_length : 15)));
                    if (literal_length >= 15) {
                        emit_length(literal_length - 15);
                    }
                    output.insert(output.end(), &source[anchor], &source[anchor] + literal_length);
                    const unsigned long long offset = index - candidate;
                    output.push_back(static_cast<unsigned char>(offset & 0xFF));
                    output.push_back(static_cast<unsigned char>((offset >> 8) & 0xFF));
                    if (match_length >= 15) {
                        emit_length(match_length - 15);
                    }
                    anchor = match_end;
                    index = match_end;
                }
                else {
                    ++index;
                }
            }
        }
        const unsigned long long literal_length = source_length - anchor;
        output.push_back(static_cast<unsigned char>((literal_length < 15 ? literal_length : 15) << 4));
        if (literal_length >= 15) {
            emit_length(literal_length - 15);
        }
        output.insert(output.end(), &source[anchor], &source[anchor] + literal_length);
    }

    // Compress into a single lz4 frame of independent blocks.
    static std::vector<unsigned char> compress_frame(const unsigned char* source, const unsigned long long source_length) {
        std::vector<unsigned char> output;
        output.push_back(0x04);
        output.push_back(0x22);
        output.push_back(0x4D);
        output.push_back(0x18);
        // Version one, independent blocks, four megabyte maximum block size.
        constexpr static const unsigned char descriptor[2] = { 0x60, 0x70 };
        output.push_back(descriptor[0]);
        output.push_back(descriptor[1]);
        output.push_back(static_cast<unsigned char>((xxh32(&descriptor[0], 2) >> 8) & 0xFF));
        constexpr static const unsigned long long block_size = 1ull << 20;
        for (unsigned long long start = 0; start < source_length; start += block_size) {
            const unsigned long long length = (source_length - start < block_size) ? (source_length - start) : block_size;
            std::vector<unsigned char> block;
            compress_block(&source[start], length, block);
            if (block.size() < length) {
                for (int i = 0; i < 4; ++i) {
                    output.push_back(static_cast<unsigned char>((block.size() >> (8 * i)) & 0xFF));
                }
                output.insert(output.end(), block.begin(), block.end());
            }
            else {
                const unsigned int stored = static_cast<unsigned int>(length) | 0x80000000u;
                for (int i = 0; i < 4; ++i) {
                    output.push_back(static_cast<unsigned char>((stored >> (8 * i)) & 0xFF));
                }
                output.insert(output.end(), &source[start], &source[start] + length);
            }
        }
        for (int i = 0; i < 4; ++i) {
            output.push_back(0); // The end mark.
        }
        return output;
    }

    // Decompress a stream of lz4 frames, appending to the output.
    static bool decompress_frame(const unsigned char* source, const unsigned long long source_length, std::vector<unsigned char>& output, std::string& error) {
        unsigned long long index = 0;
        while (index < source_length) {
            if (source_length - index < 4) {
                error = "truncated lz4 frame magic";
                return false;
            }
            const unsigned int magic = read_u32(&source[index]);
            index += 4;
            if ((magic >= skippable_magic_first) && (magic <= skippable_magic_last)) {
                if (source_length - index < 4) {
                    error = "truncated lz4 skippable frame";
                    return false;
                }
                const unsigned int skip = read_u32(&source[index]);
                index += 4;
                if (skip > source_length - index) {
                    error = "truncated lz4 skippable frame";
                    return false;
                }
                index += skip;
                continue;
            }
            if (magic != frame_magic) {
                error = "not an lz4 frame";
                return false;
            }
            if (source_length - index < 2) {
                error = "truncated lz4 frame header";
                return false;
            }
            const unsigned char flags = source[index++];
            index += 1; // The block descriptor byte, the maximum block size is not needed.
            if (((flags >> 6) & 0x03) != 0x01) {
                error = "unsupported lz4 frame version";
                return false;
            }
            if (flags & 0x01) {
                error = "lz4 dictionaries are unsupported";
                return false;
            }
            const bool block_checksums = (flags & 0x10) != 0;
            const bool content_size = (flags & 0x08) != 0;
            const bool content_checksum = (flags & 0x04) != 0;
            if (content_size) {
                if (source_length - index < 8) {
                    error = "truncated lz4 frame header";
                    return false;
                }
                index += 8;
            }
            if (source_length - index < 1) {
                error = "truncated lz4 frame header";
                return false;
            }
            index += 1; // The header checksum byte, not verified.
            while (true) {
                if (source_length - index < 4) {
                    error = "truncated lz4 block size";
                    return false;
                }
                const unsigned int block_size = read_u32(&source[index]);
                index += 4;
                if (block_size == 0) {
                    // The end mark.
                    break;
                }
                const bool uncompressed = (block_size & 0x80000000u) != 0;
                const unsigned long long length = block_size & 0x7FFFFFFFu;
                if (length > source_length - index) {
                    error = "truncated lz4 block";
                    return false;
                }
                if (uncompressed) {
                    output.insert(output.end(), &source[index], &source[index] + length);
                }
                else if (!decompress_block(&source[index], length, output, error)) {
                    return false;
                }
                index += length;
                if (block_checksums) {
                    if (source_length - index < 4) {
                        error = "truncated lz4 block checksum";
                        return false;
                    }
                    index += 4;
                }
            }
            if (content_checksum) {
                if (source_length - index < 4) {
                    error = "truncated lz4 content checksum";
                    return false;
                }
                index += 4;
            }
        }
        return true;
    }
};

#endif // ZEROSLAM_COMMON_LZ4_HPP
