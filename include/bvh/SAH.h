#ifndef BVH2_SAH_H
#define BVH2_SAH_H

#include "BVHNode.h"
#include <taskflow/taskflow.hpp>
#include <taskflow/algorithm/for_each.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace bvh2
{
namespace detail
{

template<typename S> class SAHBuilder {
    static constexpr uint32_t Bins=8;
    static constexpr uint32_t CoarseLimit=16384;
    using Box=bvh2::Box<S>;
    struct Reference { Box bounds; bvh2::Vec3<S> center; uint32_t id; };
    struct Bin { Box bounds; uint32_t count=0; };
    const std::vector<bvh2::PrimitiveT<S>>& triangles_;
    std::vector<Reference> refs_;
    BVHData<S> nodes_;
    static void expand(Box& a,const Box& b) {
        for(int axis=0;axis<3;++axis) {
            a.min[axis]=std::min(a.min[axis],b.min[axis]);
            a.max[axis]=std::max(a.max[axis],b.max[axis]);
        }
    }
    static double area(const Box& b) {
        const double x=double(b.max.x)-b.min.x,y=double(b.max.y)-b.min.y,z=double(b.max.z)-b.min.z;
        return 2*(x*y+y*z+z*x);
    }
    static uint32_t imbalance(uint32_t k,uint32_t n) { return k>n-k ? k-(n-k) : n-k-k; }
    static uint32_t bin_index(S center,S minimum,double scale) {
        return uint32_t(std::clamp((double(center)-double(minimum))*scale,0.0,double(Bins-1)));
    }
    uint32_t split_range(uint32_t first,uint32_t count,const Box& bounds,Box& leftBounds,Box& rightBounds) {
        std::array<std::array<Bin,Bins>,3> histogram;
        std::array<double,3> scale{};
        for(int axis=0;axis<3;++axis) {
            const double extent=double(bounds.max[axis])-double(bounds.min[axis]);
            if(extent>0 && std::isfinite(extent)) {
                const double value=double(Bins)/extent;
                if(std::isfinite(value)) scale[axis]=value;
            }
        }
        for(uint32_t i=first;i<first+count;++i) {
            const auto& r=refs_[i];
            for(int axis=0;axis<3;++axis) if(scale[axis]>0) {
                auto& bin=histogram[axis][bin_index(r.center[axis],bounds.min[axis],scale[axis])];
                ++bin.count; expand(bin.bounds,r.bounds);
            }
        }
        double best=std::numeric_limits<double>::infinity();
        uint32_t bestBalance=UINT32_MAX,boundary=0;
        int bestAxis=-1;
        for(int axis=0;axis<3;++axis) if(scale[axis]>0) {
            std::array<Box,Bins> prefixBounds;
            std::array<uint32_t,Bins> prefixCounts{};
            Box left; uint32_t leftCount=0;
            for(uint32_t b=0;b<Bins;++b) {
                const auto& bin=histogram[axis][b];
                if(bin.count) { expand(left,bin.bounds); leftCount+=bin.count; }
                prefixBounds[b]=left; prefixCounts[b]=leftCount;
            }
            Box right; uint32_t rightCount=0;
            for(uint32_t b=Bins-1;b>0;--b) {
                const auto& bin=histogram[axis][b];
                if(bin.count) { expand(right,bin.bounds); rightCount+=bin.count; }
                const uint32_t n=prefixCounts[b-1];
                if(!n || !rightCount) continue;
                const double cost=area(prefixBounds[b-1])*double(n)+area(right)*double(rightCount);
                const uint32_t balance=imbalance(n,count);
                if(cost<best || (cost==best && balance<bestBalance)) {
                    best=cost; bestBalance=balance; bestAxis=axis; boundary=b-1;
                    leftBounds=prefixBounds[b-1]; rightBounds=right;
                }
            }
        }
        if(bestAxis>=0) {
            const auto begin=refs_.begin()+first;
            const auto middle=std::partition(begin,begin+count,[&](const Reference& r) {
                return (double(r.center[bestAxis])-double(bounds.min[bestAxis]))*scale[bestAxis]<double(boundary+1);
            });
            const uint32_t split=uint32_t(middle-begin);
            if(split && split<count) return split;
        }
        const uint32_t split=count/2;
        leftBounds=Box{}; rightBounds=Box{};
        for(uint32_t i=0;i<count;++i) expand(i<split ? leftBounds : rightBounds,refs_[first+i].bounds);
        return split;
    }
    void build(uint32_t first,uint32_t count,uint32_t index,uint32_t next,uint32_t depth,tf::Subflow* subflow) {
        if(count==1) { nodes_.bounds[index]=refs_[first].bounds; nodes_.primitiveIndices[index]=refs_[first].id; return; }
        const uint32_t left=next,right=next+1;
        nodes_.childOffsets[index]=left;
        if(count==2) {
            nodes_.bounds[left]=refs_[first].bounds; nodes_.primitiveIndices[left]=refs_[first].id;
            nodes_.bounds[right]=refs_[first+1].bounds; nodes_.primitiveIndices[right]=refs_[first+1].id;
            return;
        }
        uint32_t split=0;
        // Limit expensive SAH evaluation to smaller ranges.
        if(count>CoarseLimit && depth<64) {
            Box centers;
            for(uint32_t i=first;i<first+count;++i) centers.expand(refs_[i].center);
            int axis=0;
            for(int a=1;a<3;++a)
                if(double(centers.max[a])-double(centers.min[a])>double(centers.max[axis])-double(centers.min[axis])) axis=a;
            const double middle=double(centers.min[axis])*0.5+double(centers.max[axis])*0.5;
            const auto begin=refs_.begin()+first;
            split=uint32_t(std::partition(begin,begin+count,[&](const Reference& r) { return double(r.center[axis])<middle; })-begin);
            if(split && split<count) {
                for(uint32_t i=0;i<count;++i) expand(nodes_.bounds[i<split ? left : right],refs_[first+i].bounds);
                // Retry SAH when the cheap split leaves almost the full parent area on both sides.
                const double cost=area(nodes_.bounds[left])*double(split)+area(nodes_.bounds[right])*double(count-split);
                if(cost>=0.9*area(nodes_.bounds[index])*double(count)) split=0;
            }
        }
        if(!split || split==count) {
            if(depth<64) split=split_range(first,count,nodes_.bounds[index],nodes_.bounds[left],nodes_.bounds[right]);
            else {
                split=count/2;
                for(uint32_t i=0;i<count;++i) expand(nodes_.bounds[i<split ? left : right],refs_[first+i].bounds);
            }
        }
        if(subflow && count>=1024) {
            subflow->emplace([=](tf::Subflow& child) { build(first,split,left,next+2,depth+1,&child); });
            subflow->emplace([=](tf::Subflow& child) { build(first+split,count-split,right,next+2*split,depth+1,&child); });
        } else {
            build(first,split,left,next+2,depth+1,nullptr);
            build(first+split,count-split,right,next+2*split,depth+1,nullptr);
        }
    }
public:
    explicit SAHBuilder(const std::vector<bvh2::PrimitiveT<S>>& triangles)
        : triangles_(triangles) {}
    BVHData<S> run(tf::Executor& executor, const std::vector<uint32_t>* order=nullptr) {
        static_assert(std::is_floating_point_v<S>, "BVH construction requires floating-point coordinates");
        const size_t n=triangles_.size();
        if(!n) return {};
        if(n>UINT32_MAX/2) throw std::runtime_error("BVH node indices exceed uint32 capacity");
        if(order && order->size()!=n) throw std::invalid_argument("Primitive ordering size mismatch");
        refs_.resize(n); nodes_.resize(2*n-1);
        tf::Taskflow flow;
        auto prepare = flow.for_each_index(std::size_t(0), n, std::size_t(1), [this, order](std::size_t i) {
            auto& r=refs_[i];
            r.id=order ? (*order)[i] : uint32_t(i);
            if(r.id>=triangles_.size()) throw std::invalid_argument("Primitive index outside input range");
            r.bounds=triangles_[r.id].bounds;
            for(int axis=0;axis<3;++axis) {
                if(!std::isfinite(r.bounds.min[axis]) || !std::isfinite(r.bounds.max[axis]) || r.bounds.min[axis]>r.bounds.max[axis])
                    throw std::runtime_error("BVH requires finite ordered primitive bounds");
                r.center[axis]=r.bounds.min[axis]*S(0.5)+r.bounds.max[axis]*S(0.5);
            }
        });
        auto construct = flow.emplace([&](tf::Subflow& sf) {
            for(const auto& r:refs_) expand(nodes_.bounds[0],r.bounds);
            build(0,uint32_t(n),0,1,0,&sf);
        });
        prepare.precede(construct);
        executor.run(flow).get();
        return std::move(nodes_);
    }
};

} // namespace detail

template<typename Scalar>
BVHData<Scalar>
buildSAHBVH(tf::Executor &executor, const std::vector<PrimitiveT<Scalar>> &primitives)
{
    return detail::SAHBuilder<Scalar>(primitives).run(executor);
}

} // namespace bvh2

#endif // BVH2_SAH_H
