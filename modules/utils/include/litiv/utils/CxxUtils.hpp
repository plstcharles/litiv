#pragma once

#if __cplusplus<201103L
#error "This project requires C++11 support."
#endif //__cplusplus<=201103L

#include <thread>
#include <mutex>
#include <chrono>
#include <atomic>
#include <future>
#include <functional>
#include <condition_variable>

namespace CxxUtils {

    template<typename Derived,typename Base,typename Del>
    std::unique_ptr<Derived,Del> static_unique_ptr_cast(std::unique_ptr<Base,Del>&& p) {
        auto d = static_cast<Derived*>(p.release());
        return std::unique_ptr<Derived,Del>(d,std::move(p.get_deleter()));
    }

    template<typename Derived,typename Base,typename Del>
    std::unique_ptr<Derived,Del>dynamic_unique_ptr_cast(std::unique_ptr<Base,Del>&& p) {
        if(Derived* result = dynamic_cast<Derived*>(p.get())) {
            p.release();
            return std::unique_ptr<Derived,Del>(result,std::move(p.get_deleter()));
        }
        return std::unique_ptr<Derived,Del>(nullptr,p.get_deleter());
    }

    template<size_t n, typename F>
    inline typename std::enable_if<n==0>::type unroll(const F& f) {
        f(0);
    }

    template<size_t n, typename F>
    inline typename std::enable_if<(n>0)>::type unroll(const F& f) {
        unroll<n-1>(f);
        f(n);
    }

    template<typename T,std::size_t nByteAlign>
    class AlignAllocator {
    public:
        typedef T value_type;
        typedef T* pointer;
        typedef T& reference;
        typedef const T* const_pointer;
        typedef const T& const_reference;
        typedef std::size_t size_type;
        typedef std::ptrdiff_t difference_type;
        typedef std::true_type propagate_on_container_move_assignment;
        template<typename T2> struct rebind {typedef AlignAllocator<T2,nByteAlign> other;};
    public:
        inline AlignAllocator() noexcept {}
        template<typename T2> inline AlignAllocator(const AlignAllocator<T2,nByteAlign>&) noexcept {}
        inline ~AlignAllocator() throw() {}
        inline pointer address(reference r) {return std::addressof(r);}
        inline const_pointer address(const_reference r) const noexcept {return std::addressof(r);}
#if PLATFORM_USES_WIN32API
        inline pointer allocate(size_type n) {
            const size_type alignment = static_cast<size_type>(nByteAlign);
            const size_type alignment = static_cast<size_type>(nByteAlign);
            size_t alloc_size = n*sizeof(value_type);
            if((alloc_size%alignment)!=0) {
                alloc_size += alignment - alloc_size%alignment;
                CV_DbgAssert((alloc_size%alignment)==0);
            }
            void* ptr = _aligned_malloc(alloc_size,nByteAlign);
            if(ptr==nullptr)
                throw std::bad_alloc();
            return reinterpret_cast<pointer>(ptr);
        }
        inline void deallocate(pointer p, size_type) noexcept {_aligned_free(p);}
#else //!PLATFORM_USES_WIN32API
        inline pointer allocate(size_type n) {
            const size_type alignment = static_cast<size_type>(nByteAlign);
            size_t alloc_size = n*sizeof(value_type);
            if((alloc_size%alignment)!=0) {
                alloc_size += alignment - alloc_size%alignment;
                CV_DbgAssert((alloc_size%alignment)==0);
            }
            void* ptr = aligned_alloc(alignment,alloc_size);
            if(ptr==nullptr)
                throw std::bad_alloc();
            return reinterpret_cast<pointer>(ptr);
        }
        inline void deallocate(pointer p, size_type) noexcept {free(p);}
#endif //!PLATFORM_USES_WIN32API
        template<class T2, class ...Args> inline void construct(T2* p, Args&&... args) {::new(reinterpret_cast<void*>(p)) T2(std::forward<Args>(args)...);}
        inline void construct(pointer p, const value_type& wert) {new(p) value_type(wert);}
        inline void destroy(pointer p) {p->~value_type();}
        inline size_type max_size() const noexcept {return (size_type(~0)-size_type(nByteAlign))/sizeof(value_type);}
        bool operator!=(const AlignAllocator<T,nByteAlign>& other) const {return !(*this==other);}
        bool operator==(const AlignAllocator<T,nByteAlign>& other) const {return true;}
    };

} //namespace CxxUtils

namespace std {
    template<typename T, size_t N>
    using aligned_vector = vector<T,CxxUtils::AlignAllocator<T,N>>;
} //namespace std
