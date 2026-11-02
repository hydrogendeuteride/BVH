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
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace bvh2
{
namespace detail
{

template<typename S> class SAHBuilder {
    using Box=bvh2::Box<S>;
    using Node=bvh2::BVHNodeT<S>;
    struct Reference { Box bounds; bvh2::Vec3<S> center; uint32_t id; };
    struct Bin { Box bounds; uint32_t count=0; };
    const std::vector<bvh2::PrimitiveT<S>>& triangles_;
    std::vector<Reference> refs_;
    std::vector<Node> nodes_;
    static void expand(Box& a,const Box& b) { a.expand(b.min); a.expand(b.max); }
    static double area(const Box& b) {
        const double x=double(b.max.x)-b.min.x,y=double(b.max.y)-b.min.y,z=double(b.max.z)-b.min.z;
        return 2*(x*y+y*z+z*x);
    }
    static uint32_t imbalance(uint32_t k,uint32_t n) { return k>n-k ? k-(n-k) : n-k-k; }
    static uint32_t bin_index(S center,S minimum,S maximum,double scale,uint32_t bins) {
        if(center==minimum) return 0;
        if(center==maximum) return bins-1;
        const double position=(double(center)-double(minimum))*scale;
        return 1+uint32_t(std::clamp(position,0.0,double(bins-3)));
    }
    uint32_t small_split(uint32_t first,uint32_t count) {
        double best_cost=std::numeric_limits<double>::infinity();
        uint32_t best_balance=UINT32_MAX,split=count/2;
        std::array<uint32_t,8> best_order{},order{};
        std::iota(best_order.begin(),best_order.begin()+count,0u);
        std::array<double,8> prefix{};
        for(int axis=0;axis<3;++axis) {
            std::iota(order.begin(),order.begin()+count,0u);
            auto less=[&](uint32_t a,uint32_t b) {
                const auto& ra=refs_[first+a]; const auto& rb=refs_[first+b];
                return ra.center[axis]==rb.center[axis] ? ra.id<rb.id : ra.center[axis]<rb.center[axis];
            };
            for(uint32_t i=1;i<count;++i) {
                const uint32_t value=order[i]; uint32_t j=i;
                while(j && less(value,order[j-1])) { order[j]=order[j-1]; --j; }
                order[j]=value;
            }
            Box left;
            for(uint32_t k=0;k<count-1;++k) {
                expand(left,refs_[first+order[k]].bounds); prefix[k]=area(left)*double(k+1);
            }
            Box right;
            for(uint32_t k=count-1;k>0;--k) {
                expand(right,refs_[first+order[k]].bounds);
                const double cost=prefix[k-1]+area(right)*double(count-k);
                const uint32_t balance=imbalance(k,count);
                if(cost<best_cost || (cost==best_cost && balance<best_balance)) {
                    best_cost=cost; best_balance=balance; split=k; best_order=order;
                }
            }
        }
        std::array<Reference,8> sorted;
        for(uint32_t i=0;i<count;++i) sorted[i]=refs_[first+best_order[i]];
        std::copy_n(sorted.begin(),count,refs_.begin()+first);
        return split;
    }
    uint32_t binned_split(uint32_t first,uint32_t count,const Box& centers) {
        constexpr uint32_t bins=18; // 16 interior bins plus the two endpoints.
        std::array<std::array<Bin,bins>,3> histogram;
        std::array<double,3> scale{};
        for(int axis=0;axis<3;++axis) {
            const double extent=double(centers.max[axis])-double(centers.min[axis]);
            if(extent>0 && std::isfinite(extent)) {
                const double candidate=double(bins-2)/extent;
                if(std::isfinite(candidate)) scale[axis]=candidate;
            }
        }
        for(uint32_t i=first;i<first+count;++i) {
            const auto& r=refs_[i];
            for(int axis=0;axis<3;++axis) if(scale[axis]>0) {
                auto& bin=histogram[axis][bin_index(r.center[axis],centers.min[axis],centers.max[axis],scale[axis],bins)];
                ++bin.count; expand(bin.bounds,r.bounds);
            }
        }
        double best_cost=std::numeric_limits<double>::infinity();
        uint32_t best_balance=UINT32_MAX,boundary=0;
        int best_axis=-1;
        for(int axis=0;axis<3;++axis) if(scale[axis]>0) {
            std::array<double,bins> prefix{};
            std::array<uint32_t,bins> left_counts{};
            Box left; uint32_t left_count=0;
            for(uint32_t b=0;b<bins;++b) {
                const auto& bin=histogram[axis][b];
                if(bin.count) { expand(left,bin.bounds); left_count+=bin.count; }
                left_counts[b]=left_count;
                prefix[b]=left_count ? area(left)*double(left_count) : 0;
            }
            Box right; uint32_t right_count=0;
            for(uint32_t b=bins-1;b>0;--b) {
                const auto& bin=histogram[axis][b];
                if(bin.count) { expand(right,bin.bounds); right_count+=bin.count; }
                if(!left_counts[b-1] || !right_count) continue;
                const double cost=prefix[b-1]+area(right)*double(right_count);
                const uint32_t balance=imbalance(left_counts[b-1],count);
                if(cost<best_cost || (cost==best_cost && balance<best_balance)) {
                    best_cost=cost; best_balance=balance; best_axis=axis; boundary=b-1;
                }
            }
        }
        if(best_axis<0) return count/2;
        const auto begin=refs_.begin()+first;
        const auto middle=std::partition(begin,begin+count,[&](const Reference& r) {
            return bin_index(r.center[best_axis],centers.min[best_axis],centers.max[best_axis],scale[best_axis],bins)<=boundary;
        });
        const uint32_t split=uint32_t(middle-begin);
        return split && split<count ? split : count/2;
    }
    void build(uint32_t first,uint32_t count,uint32_t index,uint32_t next,uint32_t depth,tf::Subflow* subflow) {
        auto& node=nodes_[index];
        if(count==1) { node.bounds=refs_[first].bounds; node.object_idx=refs_[first].id; return; }
        Box centers;
        for(uint32_t i=first;i<first+count;++i) {
            expand(node.bounds,refs_[i].bounds); centers.expand(refs_[i].center);
        }
        uint32_t split=count/2;
        if(count>2 && depth<64 && area(node.bounds)>0)
            split=count<=8 ? small_split(first,count) : binned_split(first,count,centers);
        const uint32_t left=next;
        const uint32_t right=left+1;
        node.left_idx=left;
        // Each subtree owns a disjoint range of node slots.
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
    std::vector<Node> run(tf::Executor& executor, const std::vector<uint32_t>* order=nullptr) {
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
            build(0,uint32_t(n),0,1,0,&sf);
        });
        prepare.precede(construct);
        executor.run(flow).get();
        return std::move(nodes_);
    }
};

} // namespace detail

template<typename Scalar>
std::vector<BVHNodeT<Scalar>>
buildSAHBVH(tf::Executor &executor, const std::vector<PrimitiveT<Scalar>> &primitives)
{
    return detail::SAHBuilder<Scalar>(primitives).run(executor);
}

} // namespace bvh2

#endif // BVH2_SAH_H
