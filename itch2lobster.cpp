// itch2lobster.cpp - single-symbol NASDAQ TotalView-ITCH 5.0 -> LOBSTER-format order book.
//
// Build:   g++ -O3 -march=native -std=c++17 -o itch2lobster itch2lobster.cpp -lz
//          (Debian/Ubuntu needs zlib headers: sudo apt install zlib1g-dev)
// Run:     ./itch2lobster --input 01302019.NASDAQ_ITCH50.gz --output AAPL_2019-01-30_orderbook_10.csv --symbol AAPL --levels 10
//          zcat file.gz | ./itch2lobster --input - --output out.csv --symbol AAPL
//
// Output (no header, one row per book-changing event, like LOBSTER's orderbook file):
//   AskPrice1,AskSize1,BidPrice1,BidSize1, ... , AskPriceK,AskSizeK,BidPriceK,BidSizeK
// Prices are integers = dollars x 10,000 (ITCH's native 4-decimal format == LOBSTER's scale).
// Empty levels: ask price -9999999999 / bid price 9999999999 with size 0 (same convention as the
// Python reconstructor this replaces; downstream code treats |price| >= 999,999,000 as "empty").
//
// The whole-day book is maintained from the first message, but rows are only written inside
// the window (default 09:30:00-16:00:00 ET). Use --all-day to write every event.
//
// Input framing: .NASDAQ_ITCH50 files are a stream of [2-byte big-endian length][message].
// Handled message types: R (stock directory), A/F (add), E/C (execute), X (cancel), D (delete),
// U (replace = atomic: ONE row after delete+add). Everything else is skipped.

#include <zlib.h>

#include <charconv>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

static inline uint16_t be16(const uint8_t* p) { return (uint16_t)((p[0] << 8) | p[1]); }
static inline uint32_t be32(const uint8_t* p) {
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) | ((uint32_t)p[2] << 8) | p[3];
}
static inline uint64_t be48(const uint8_t* p) {
    return ((uint64_t)p[0] << 40) | ((uint64_t)p[1] << 32) | ((uint64_t)p[2] << 24) |
           ((uint64_t)p[3] << 16) | ((uint64_t)p[4] << 8) | p[5];
}
static inline uint64_t be64(const uint8_t* p) {
    return ((uint64_t)be32(p) << 32) | be32(p + 4);
}

// ---------------------------------------------------------------- buffered gz reader
struct Reader {
    gzFile f = nullptr;
    std::vector<uint8_t> buf;
    size_t pos = 0, end = 0;
    bool eof = false;
    bool stream_error = false;
    explicit Reader(size_t cap = 1u << 24) : buf(cap) {}
    // Guarantee n readable bytes at buf[pos]; false if the stream ends first.
    bool ensure(size_t n) {
        if (end - pos >= n) return true;
        if (eof) return false;
        if (pos) {
            memmove(buf.data(), buf.data() + pos, end - pos);
            end -= pos;
            pos = 0;
        }
        while (end < buf.size() && !eof) {
            int r = gzread(f, buf.data() + end, (unsigned)(buf.size() - end));
            if (r < 0) {
                int errnum;
                const char* msg = gzerror(f, &errnum);
                fprintf(stderr, "gzread error: %s\n", msg ? msg : "?");
                exit(3);
            }
            if (r == 0) {
                eof = true;
                int errnum = Z_OK;
                const char* msg = gzerror(f, &errnum);
                if (errnum != Z_OK && errnum != Z_STREAM_END) {
                    fprintf(stderr, "ERROR: input is truncated or corrupt (%s). Re-download it.\n",
                            msg ? msg : "gz error");
                    stream_error = true;
                }
            } else end += (size_t)r;
        }
        return end - pos >= n;
    }
};

// ---------------------------------------------------------------- order book
struct Order {
    uint32_t price;
    uint32_t shares;
    bool buy;
};

struct Book {
    std::unordered_map<uint64_t, Order> orders;
    std::map<uint32_t, int64_t, std::greater<uint32_t>> bids;
    std::map<uint32_t, int64_t> asks;

    void add(uint64_t ref, bool buy, uint32_t shares, uint32_t price) {
        orders[ref] = Order{price, shares, buy};
        (buy ? bids[price] : asks[price]) += shares;
    }
    template <class M>
    static void sub(M& m, uint32_t price, uint32_t n) {
        auto it = m.find(price);
        if (it == m.end()) return;
        it->second -= n;
        if (it->second <= 0) m.erase(it);
    }
    // Remove up to n shares of an order. Returns false if the order is unknown.
    bool reduce(uint64_t ref, uint32_t n) {
        auto it = orders.find(ref);
        if (it == orders.end()) return false;
        Order& o = it->second;
        if (n > o.shares) n = o.shares;
        if (o.buy) sub(bids, o.price, n);
        else sub(asks, o.price, n);
        o.shares -= n;
        if (o.shares == 0) orders.erase(it);
        return true;
    }
    bool remove(uint64_t ref) {
        auto it = orders.find(ref);
        if (it == orders.end()) return false;
        return reduce(ref, it->second.shares);
    }
    bool replace(uint64_t old_ref, uint64_t new_ref, uint32_t shares, uint32_t price) {
        auto it = orders.find(old_ref);
        if (it == orders.end()) return false;
        bool buy = it->second.buy;
        remove(old_ref);
        add(new_ref, buy, shares, price);
        return true;
    }
};

// ---------------------------------------------------------------- row output
struct Writer {
    FILE* out = nullptr;
    FILE* times = nullptr;
    std::vector<char> buf;
    int levels = 10;
    uint64_t rows = 0;
    Writer() { buf.reserve(1 << 20); }

    void put(int64_t v) {
        char tmp[24];
        auto r = std::to_chars(tmp, tmp + sizeof tmp, v);
        buf.insert(buf.end(), tmp, r.ptr);
    }
    void row(const Book& b, uint64_t ts_ns) {
        auto ia = b.asks.begin();
        auto ib = b.bids.begin();
        for (int l = 0; l < levels; l++) {
            if (l) buf.push_back(',');
            if (ia != b.asks.end()) { put(ia->first); buf.push_back(','); put(ia->second); ++ia; }
            else { put(-9999999999LL); buf.push_back(','); put(0); }
            buf.push_back(',');
            if (ib != b.bids.end()) { put(ib->first); buf.push_back(','); put(ib->second); ++ib; }
            else { put(9999999999LL); buf.push_back(','); put(0); }
        }
        buf.push_back('\n');
        if (times) fprintf(times, "%llu\n", (unsigned long long)ts_ns);
        rows++;
        if (buf.size() > (1u << 20) - 4096) flush();
    }
    void flush() {
        if (!buf.empty()) fwrite(buf.data(), 1, buf.size(), out);
        buf.clear();
    }
};

static void usage() {
    fprintf(stderr,
            "usage: itch2lobster --input FILE|- --output FILE|- --symbol AAPL [--levels 10]\n"
            "       [--start-sec 34200] [--end-sec 57600] [--all-day] [--max-messages N]\n"
            "       [--progress-every N] [--times FILE]\n");
}

int main(int argc, char** argv) {
    std::string in, outp, sym, times_path;
    int levels = 10;
    double start_sec = 34200.0, end_sec = 57600.0;
    bool all_day = false;
    uint64_t max_msgs = 0, progress = 100000000ULL;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) { usage(); exit(1); }
            return argv[++i];
        };
        if (a == "--input") in = next();
        else if (a == "--output") outp = next();
        else if (a == "--symbol") sym = next();
        else if (a == "--levels") levels = atoi(next());
        else if (a == "--start-sec") start_sec = atof(next());
        else if (a == "--end-sec") end_sec = atof(next());
        else if (a == "--all-day") all_day = true;
        else if (a == "--max-messages") max_msgs = strtoull(next(), nullptr, 10);
        else if (a == "--progress-every") progress = strtoull(next(), nullptr, 10);
        else if (a == "--times") times_path = next();
        else if (a == "-h" || a == "--help") { usage(); return 0; }
        else { fprintf(stderr, "unknown argument: %s\n", a.c_str()); usage(); return 1; }
    }
    if (in.empty() || outp.empty() || sym.empty() || levels < 1 || levels > 50) { usage(); return 1; }

    char sym8[9];
    memset(sym8, ' ', 8);
    sym8[8] = 0;
    if (sym.size() > 8) { fprintf(stderr, "symbol too long\n"); return 1; }
    memcpy(sym8, sym.data(), sym.size());

    Reader rd;
    rd.f = (in == "-") ? gzdopen(0, "rb") : gzopen(in.c_str(), "rb");
    if (!rd.f) { fprintf(stderr, "cannot open %s\n", in.c_str()); return 1; }

    Writer w;
    w.levels = levels;
    w.out = (outp == "-") ? stdout : fopen(outp.c_str(), "wb");
    if (!w.out) { fprintf(stderr, "cannot write %s\n", outp.c_str()); return 1; }
    if (!times_path.empty()) w.times = fopen(times_path.c_str(), "w");

    const uint64_t t_start = (uint64_t)(start_sec * 1e9), t_end = (uint64_t)(end_sec * 1e9);
    Book book;
    book.orders.reserve(1 << 20);
    int target = -1;
    uint64_t nmsg = 0, n_sym = 0, n_unknown_ref = 0, n_malformed = 0;

    while (rd.ensure(2)) {
        uint16_t len = be16(&rd.buf[rd.pos]);
        if (!rd.ensure(2u + len)) { fprintf(stderr, "warning: truncated final message\n"); break; }
        const uint8_t* m = &rd.buf[rd.pos + 2];
        rd.pos += 2u + len;
        nmsg++;
        if (len == 0) { n_malformed++; continue; }
        if (max_msgs && nmsg > max_msgs) break;
        if (progress && nmsg % progress == 0)
            fprintf(stderr, "  %llu M messages, %llu symbol msgs, %llu rows\n",
                    (unsigned long long)(nmsg / 1000000), (unsigned long long)n_sym,
                    (unsigned long long)w.rows);

        const uint8_t type = m[0];
        if (type == 'R') {
            if (len >= 19 && target < 0 && memcmp(m + 11, sym8, 8) == 0) {
                target = be16(m + 1);
                fprintf(stderr, "symbol %s -> stock locate %d\n", sym.c_str(), target);
            }
            continue;
        }
        if (target < 0 || len < 3 || be16(m + 1) != (uint16_t)target) continue;

        bool changed = false;
        switch (type) {
            case 'A':
            case 'F':
                if (len < 36) { n_malformed++; break; }
                book.add(be64(m + 11), m[19] == 'B', be32(m + 20), be32(m + 32));
                changed = true;
                break;
            case 'E':
            case 'C':
                if (len < 31) { n_malformed++; break; }
                changed = book.reduce(be64(m + 11), be32(m + 19));
                if (!changed) n_unknown_ref++;
                break;
            case 'X':
                if (len < 23) { n_malformed++; break; }
                changed = book.reduce(be64(m + 11), be32(m + 19));
                if (!changed) n_unknown_ref++;
                break;
            case 'D':
                if (len < 19) { n_malformed++; break; }
                changed = book.remove(be64(m + 11));
                if (!changed) n_unknown_ref++;
                break;
            case 'U':
                if (len < 35) { n_malformed++; break; }
                changed = book.replace(be64(m + 11), be64(m + 19), be32(m + 27), be32(m + 31));
                if (!changed) n_unknown_ref++;
                break;
            default:
                break;
        }
        if (!changed) continue;
        n_sym++;
        uint64_t ts = be48(m + 5);
        if (all_day || (ts >= t_start && ts <= t_end)) w.row(book, ts);
    }

    w.flush();
    if (w.out != stdout) fclose(w.out);
    if (w.times) fclose(w.times);
    gzclose(rd.f);

    fprintf(stderr, "Done. %llu messages read, %llu changed the %s book, %llu rows written",
            (unsigned long long)nmsg, (unsigned long long)n_sym, sym.c_str(), (unsigned long long)w.rows);
    fprintf(stderr, " (unknown order refs: %llu, malformed: %llu)\n",
            (unsigned long long)n_unknown_ref, (unsigned long long)n_malformed);
    if (rd.stream_error) return 4;
    if (target < 0) {
        fprintf(stderr, "ERROR: symbol %s not found in any stock-directory (R) message.\n", sym.c_str());
        return 2;
    }
    if (w.rows == 0)
        fprintf(stderr, "WARNING: no rows written. If this was a --max-messages smoke test the early "
                        "messages are pre-market; add --all-day.\n");
    return 0;
}
