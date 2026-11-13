// Reuse the frozen reader and collective cost formulas without editing the kernel.
#define main frozen_orbit_main
#include "fast.cpp"
#undef main

int main(int argc,char** argv){
 try {
    if(argc!=4)throw runtime_error("graph_priorities graph.orb network.txt output.tsv");
    Graph g=read_graph(argv[1]);Network net(argv[2],g.ranks,"ring");
    vector<double> cost(g.n),tail(g.n);map<tuple<int,uint64_t,uint64_t>,double> cache;
    for(uint32_t i=0;i<g.n;i++){
        auto& n=g.nodes[i];cost[i]=n.duration;
        if(n.kind==2){auto key=make_tuple(n.comm_type,n.members,n.bytes);auto it=cache.find(key);
            if(it==cache.end())it=cache.emplace(key,net.plan(n).seconds).first;cost[i]=it->second;}
    }
    vector<uint32_t> order;order.reserve(g.n);
    for(uint32_t i=0;i<g.n;i++)if(g.indeg[i]==0)order.push_back(i);
    for(size_t at=0;at<order.size();at++)for(uint64_t j=g.offsets[order[at]];j<g.offsets[order[at]+1];j++)
        if(--g.indeg[g.children[j]]==0)order.push_back(g.children[j]);
    if(order.size()!=g.n)throw runtime_error("Non-DAG input");
    for(auto it=order.rbegin();it!=order.rend();++it){
        uint32_t i=*it;double next=0;
        for(uint64_t j=g.offsets[i];j<g.offsets[i+1];j++)next=max(next,tail[g.children[j]]);
        tail[i]=cost[i]+next;
    }
    ofstream f(argv[3]);f<<setprecision(17);
    for(uint32_t i=0;i<g.n;i++)if(g.nodes[i].kind==2)f<<i<<'\t'<<cost[i]<<'\t'<<tail[i]<<'\n';
    if(!f)throw runtime_error("Cannot write priorities");
    cout<<"validated_dag_nodes="<<order.size()<<"\n";return 0;
 }catch(const exception& e){cerr<<e.what()<<"\n";return 1;}
}
