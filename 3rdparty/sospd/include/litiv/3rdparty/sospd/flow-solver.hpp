#ifndef _FLOW_SOLVER_HPP_
#define _FLOW_SOLVER_HPP_

#include "litiv/3rdparty/sospd/sos-graph.hpp"

class SubmodularIBFS;
struct SubmodularIBFSParams;

class FlowSolver {
    public:
        FlowSolver() = default;
        virtual ~FlowSolver() { }

        static std::unique_ptr<FlowSolver> GetSolver(const SubmodularIBFSParams& params);

        virtual void Solve(SubmodularIBFS* energy) = 0;

    private:
        // Make non-copyable, non-movable
        FlowSolver(const FlowSolver&) = delete;
        FlowSolver(FlowSolver&&) = delete;
        FlowSolver& operator=(const FlowSolver&) = delete;
        FlowSolver& operator=(FlowSolver&&) = delete;
};

class BidirectionalIBFS : public FlowSolver {
    public:
        BidirectionalIBFS() { }
        virtual ~BidirectionalIBFS() = default;

        virtual void Solve(SubmodularIBFS* energy);

        void IBFS();
        void ComputeMinCut();

    private:
        // Typedefs
        typedef SoSGraph::NodeId NodeId;
        typedef SoSGraph::CliqueId CliqueId;
        typedef SoSGraph::Node Node;
        typedef SoSGraph::NodeState NodeState;
        typedef SoSGraph::NeighborList NeighborList;
        typedef SoSGraph::ArcIterator ArcIterator;
        typedef SoSGraph::NodeQueue NodeQueue;
        typedef SoSGraph::OrphanList OrphanList;
        typedef SoSGraph::CliqueVec CliqueVec;

        // Helper functions
        void Push(ArcIterator& arc, bool forwardArc, REAL delta);
        void Augment(ArcIterator& arc);
        void Adopt();
        void MakeOrphan(NodeId i);
        void RemoveFromLayer(NodeId i);
        void AddToLayer(NodeId i);
        void AdvanceSearchNode();

        void IBFSInit();

        /* Algorithm data */ 

        SoSGraph* m_graph;
        SubmodularIBFS* m_energy;
        // Layers store vertices by distance.
        std::vector<NodeQueue> m_source_layers;
        std::vector<NodeQueue> m_sink_layers;
        OrphanList m_source_orphans;
        OrphanList m_sink_orphans;
        int m_source_tree_d;
        int m_sink_tree_d;
        typedef typename NodeQueue::iterator queue_iterator;
        queue_iterator m_search_node_iter;
        queue_iterator m_search_node_end;
        ArcIterator m_search_arc;
        ArcIterator m_search_arc_end;
        bool m_forward_search;

        // Statistics

        double m_totalTime = 0;
        double m_initTime = 0;
        double m_augmentTime = 0;
        double m_adoptTime = 0;
        size_t m_num_clique_pushes = 0;
};


class SourceIBFS : public FlowSolver {
    public:
        SourceIBFS() { }
        virtual ~SourceIBFS() = default;

        virtual void Solve(SubmodularIBFS* energy);

        void IBFS();
        void ComputeMinCut();

    protected:
        // Typedefs
        typedef SoSGraph::NodeId NodeId;
        typedef SoSGraph::CliqueId CliqueId;
        typedef SoSGraph::Node Node;
        typedef SoSGraph::NodeState NodeState;
        typedef SoSGraph::NeighborList NeighborList;
        typedef SoSGraph::ArcIterator ArcIterator;
        typedef SoSGraph::NodeQueue NodeQueue;
        typedef SoSGraph::OrphanList OrphanList;
        typedef SoSGraph::CliqueVec CliqueVec;

        // Helper functions
        void Push(ArcIterator& arc, bool forwardArc, REAL delta);
        void Augment(ArcIterator& arc);
        void Adopt();
        void MakeOrphan(NodeId i);
        void RemoveFromLayer(NodeId i);
        void AddToLayer(NodeId i);
        void AdvanceSearchNode();

        void IBFSInit();

        /* Algorithm data */ 

        SoSGraph* m_graph;
        SubmodularIBFS* m_energy;
        // Layers store vertices by distance.
        std::vector<NodeQueue> m_source_layers;
        OrphanList m_source_orphans;
        int m_source_tree_d;
        typedef typename NodeQueue::iterator queue_iterator;
        queue_iterator m_search_node_iter;
        queue_iterator m_search_node_end;
        ArcIterator m_search_arc;
        ArcIterator m_search_arc_end;

        /* Statistics */

        double m_totalTime = 0;
        double m_initTime = 0;
        double m_augmentTime = 0;
        double m_adoptTime = 0;
        size_t m_num_clique_pushes = 0;
};

class ParametricIBFS : public FlowSolver {
    public:
        ParametricIBFS() { }
        virtual ~ParametricIBFS() = default;

        virtual void Solve(SubmodularIBFS* energy);

        void IBFS();
        void ComputeMinCut();

    protected:
        // Typedefs
        typedef SoSGraph::NodeId NodeId;
        typedef SoSGraph::CliqueId CliqueId;
        typedef SoSGraph::Node Node;
        typedef SoSGraph::NodeState NodeState;
        typedef SoSGraph::NeighborList NeighborList;
        typedef SoSGraph::ArcIterator ArcIterator;
        typedef SoSGraph::NodeQueue NodeQueue;
        typedef SoSGraph::OrphanList OrphanList;
        typedef SoSGraph::CliqueVec CliqueVec;

        // Helper functions
        void Push(ArcIterator& arc, bool forwardArc, REAL delta);
        void Augment(ArcIterator& arc);
        void Adopt();
        void MakeOrphan(NodeId i);
        void RemoveFromLayer(NodeId i);
        void AddToLayer(NodeId i);
        void AdvanceSearchNode();

        void IBFSInit();

        /* Algorithm data */ 

        SoSGraph* m_graph;
        SubmodularIBFS* m_energy;
        // Layers store vertices by distance.
        std::vector<NodeQueue> m_source_layers;
        OrphanList m_source_orphans;
        int m_source_tree_d;
        typedef typename NodeQueue::iterator queue_iterator;
        queue_iterator m_search_node_iter;
        queue_iterator m_search_node_end;
        ArcIterator m_search_arc;
        ArcIterator m_search_arc_end;

        /* Statistics */

        double m_totalTime = 0;
        double m_initTime = 0;
        double m_augmentTime = 0;
        double m_adoptTime = 0;
        size_t m_num_clique_pushes = 0;

        /* Parametric flow data */

        std::vector<REAL> m_orig_c_si;
        std::vector<REAL> m_orig_c_it;
        std::vector<REAL> m_parametricUnaries;
};

#endif
