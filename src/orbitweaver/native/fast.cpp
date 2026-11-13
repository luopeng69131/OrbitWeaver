// Minimal event simulator for a fixed measured execution graph.
// Collective traffic uses explicitly synchronized algorithm rounds and routed links.
// CPU/GPU work, dependency progress and optical resource reservations are replayed.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <queue>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>
#include <sys/resource.h>
using namespace std;
struct Node { double duration; uint64_t bytes, members; uint32_t refs; uint16_t comm_type; uint8_t kind, rank; };
static_assert(sizeof(Node)==32,"Graph format requires 32-byte records");
struct Link {int src,dst;double bw,lat;};
struct Plan {double seconds=0; vector<int> resources; vector<double> bytes;};
struct Graph {uint32_t n=0,e=0,ranks=0,original=0;vector<Node> nodes;vector<uint32_t> indeg,children;vector<uint64_t> offsets;};
template<class T> void read_vec(ifstream& f,vector<T>& v,size_t n){v.resize(n);f.read(reinterpret_cast<char*>(v.data()),n*sizeof(T));if(!f)throw runtime_error("truncated graph");}
Graph read_graph(const string& path){
    ifstream f(path,ios::binary);char magic[8];f.read(magic,8);if(string(magic,8)!="ORBET001")throw runtime_error("bad graph magic");Graph g;
    for(auto p:{&g.n,&g.e,&g.ranks,&g.original})f.read(reinterpret_cast<char*>(p),4);
    if(!f||!g.ranks||g.ranks>64||g.n>100000000)throw runtime_error("invalid graph dimensions");
    read_vec(f,g.nodes,g.n);read_vec(f,g.indeg,g.n);read_vec(f,g.offsets,size_t(g.n)+1);read_vec(f,g.children,g.e);
    if(g.offsets.back()!=g.e)throw runtime_error("invalid edges");
    uint64_t refs=0;for(const auto& n:g.nodes){refs+=n.refs;if(n.kind>3||n.rank>=g.ranks||n.duration<0||!isfinite(n.duration))throw runtime_error("invalid node");}
    if(refs!=g.original)throw runtime_error("original node count mismatch");
    return g;
}
class Network {
public:
    string alltoall;int ranks;vector<Link> links;vector<int> placement;vector<vector<vector<int>>> paths;
    Network(const string& file,int expected,const string& algorithm):alltoall(algorithm){
        if(alltoall!="ring"&&alltoall!="pairwise")throw runtime_error("invalid alltoall algorithm");
        ifstream f(file);int edge_count;f>>ranks>>edge_count;if(!f||ranks!=expected||edge_count<1)throw runtime_error("invalid network");
        placement.resize(ranks);set<int> used;for(auto& p:placement){f>>p;used.insert(p);}
        if(int(used.size())!=ranks||*used.begin()!=0||*used.rbegin()!=ranks-1)throw runtime_error("placement not a permutation");
        vector<vector<int>> adj(ranks);
        for(int i=0;i<edge_count;i++){Link l;f>>l.src>>l.dst>>l.bw>>l.lat;if(!f||l.src<0||l.src>=ranks||l.dst<0||l.dst>=ranks||l.bw<=0||l.lat<0||!isfinite(l.bw)||!isfinite(l.lat))throw runtime_error("invalid link");adj[l.src].push_back(i);links.push_back(l);}
        // Deterministic shortest-hop forwarding; no learned or policy-dependent routes.
        paths.resize(ranks,vector<vector<int>>(ranks));
        for(int src=0;src<ranks;src++){
            vector<int> pred(ranks,-1);queue<int> q;q.push(src);pred[src]=-2;
            while(!q.empty()){int v=q.front();q.pop();for(int e:adj[v])if(pred[links[e].dst]==-1){pred[links[e].dst]=e;q.push(links[e].dst);}}
            for(int dst=0;dst<ranks;dst++){if(pred[dst]==-1)throw runtime_error("disconnected network");int cur=dst;while(cur!=src){int e=pred[cur];paths[src][dst].push_back(e);cur=links[e].src;}reverse(paths[src][dst].begin(),paths[src][dst].end());}
        }
    }
    Plan plan(const Node& node) const {
        Plan p;p.bytes.assign(links.size(),0);vector<int> members;
        for(int r=0;r<ranks;r++)if(node.members&(1ULL<<r)){members.push_back(r);p.resources.push_back(2*ranks+r);}
        if(members.empty())throw runtime_error("empty collective");
        int n=members.size();
        if(node.comm_type==9)return p; // All participants have reached this zero-duration barrier.
        auto round=[&](const vector<tuple<int,int,double>>& flows,int repeats=1){
            vector<double> load(links.size(),0);double longest_lat=0;
            for(auto [a,b,bytes]:flows){double lat=0;for(int edge:paths[placement[a]][placement[b]]){load[edge]+=bytes;lat+=links[edge].lat;}longest_lat=max(longest_lat,lat);}
            double serialization=0;for(size_t i=0;i<load.size();i++){serialization=max(serialization,load[i]/links[i].bw);p.bytes[i]+=load[i]*repeats;}
            p.seconds+=repeats*(serialization+longest_lat);
        };
        if(n>1){
            double s=node.bytes;vector<tuple<int,int,double>> flows;
            if(node.comm_type==0||node.comm_type==2||node.comm_type==7){
                double per_rank=node.comm_type==2?s:s/n;
                for(int i=0;i<n;i++)flows.emplace_back(members[i],members[(i+1)%n],per_rank);
                round(flows,(n-1)*(node.comm_type==0?2:1));
            }else if(node.comm_type==6){
                if(alltoall=="ring"){
                    // ASTRA native Ring.cc: n(n-1)/2 steps of S/n along the logical ring.
                    for(int i=0;i<n;i++)flows.emplace_back(members[i],members[(i+1)%n],s/n);
                    round(flows,n*(n-1)/2);
                }else{
                    for(int step=1;step<n;step++){flows.clear();for(int i=0;i<n;i++)flows.emplace_back(members[i],members[(i+step)%n],s/n);round(flows);}
                }
            }else if(node.comm_type==5){
                // Serial ring broadcast, root = lowest logical rank (explicit simplification).
                for(int i=0;i<n-1;i++)round({{members[i],members[i+1],s}});
            }else throw runtime_error("unsupported collective type");
        }
        for(size_t i=0;i<p.bytes.size();i++)if(p.bytes[i]>0)p.resources.push_back(3*ranks+i);
        sort(p.resources.begin(),p.resources.end());return p;
    }
};
struct Event {double end;uint32_t id;bool operator>(const Event& o)const{return tie(end,id)>tie(o.end,o.id);}};
int main(int argc,char** argv){
 try {
    if(argc!=4&&argc!=5)throw runtime_error("usage: orbit_engine graph.orb network.txt result.json [ring|pairwise]");
    auto wall_start=chrono::steady_clock::now();Graph g=read_graph(argv[1]);Network net(argv[2],g.ranks,argc==5?argv[4]:"ring");
    vector<double> priorities(g.n,0.0);
    if(const char* priority_file=std::getenv("ORBIT_PRIORITY_FILE")){
        ifstream f(priority_file);if(!f)throw runtime_error("priority file missing");
        uint32_t id;double score;set<uint32_t> seen;
        while(f>>id>>score){if(id>=g.n||g.nodes[id].kind!=2||!isfinite(score)||!seen.insert(id).second)throw runtime_error("invalid priority binding");priorities[id]=score;}
        if(!f.eof())throw runtime_error("malformed priority file");
    }
    vector<double> release_at(g.ranks,0.0);
    if(const char* raw=std::getenv("ORBIT_RANK_RELEASE_NS")) {
        std::stringstream in(raw);string item;size_t rank=0;
        while(std::getline(in,item,',')) {
            size_t count=0;uint64_t value=std::stoull(item,&count);
            if(rank>=g.ranks || count!=item.size() || value>1000000000000ULL)throw runtime_error("Invalid rank release");
            release_at[rank++]=value/1e9;
        }
        if(rank!=g.ranks)throw runtime_error("Rank release count mismatch");
    }
    vector<vector<uint32_t>> delayed_roots(g.ranks);
    ofstream event_log;
    if(const char* events_file=std::getenv("ORBIT_EVENT_LOG")){event_log.open(events_file);if(!event_log)throw runtime_error("event log unavailable");}
    auto log_event=[&](const char* event,uint32_t id,double at,int resource=-1,int64_t blocker=-1){
        if(event_log.is_open() && g.nodes[id].kind==2)event_log<<setprecision(15)<<"{\"event\":\""<<event<<"\",\"node\":"<<id<<",\"time_s\":"<<at<<",\"resource\":"<<resource<<",\"blocker\":"<<blocker<<"}\n";
    };
    vector<Plan> plans;vector<int> plan_ids(g.n,-1);map<tuple<int,uint64_t,uint64_t>,int> cache;
    for(uint32_t i=0;i<g.n;i++)if(g.nodes[i].kind==2){auto& n=g.nodes[i];auto key=make_tuple(n.comm_type,n.members,n.bytes);auto it=cache.find(key);if(it==cache.end()){int k=plans.size();plans.push_back(net.plan(n));cache[key]=k;plan_ids[i]=k;}else plan_ids[i]=it->second;}
    const int nr=3*g.ranks+net.links.size();vector<bool> busy(nr,false);vector<int64_t> owner(nr,-1);
    vector<uint8_t> state(g.n,0);vector<double> ready_at(g.n,0),rank_done(g.ranks,0),gpu_busy(g.ranks,0),cpu_busy(g.ranks,0),comm_busy(g.ranks,0),link_reserved(net.links.size(),0),link_bytes(net.links.size(),0);
    array<double,4> wait_seconds{};uint64_t done=0,original_done=0,coll_done=0;double now=0;
    auto resources=[&](uint32_t id)->vector<int>{const auto& n=g.nodes[id];if(n.kind==2)return plans[plan_ids[id]].resources;if(n.kind==3)return {};return {int(n.rank+(n.kind==1?g.ranks:0))};};
    map<vector<int>,int> group_index;vector<vector<int>> group_resources;vector<int> node_group(g.n);
    for(uint32_t i=0;i<g.n;i++){
        auto rs=resources(i);auto it=group_index.find(rs);
        if(it==group_index.end()){int id=group_resources.size();group_resources.push_back(rs);it=group_index.emplace(rs,id).first;}
        node_group[i]=it->second;
    }
    using ReadyQueue=priority_queue<pair<double,uint32_t>,vector<pair<double,uint32_t>>,greater<pair<double,uint32_t>>>;
    vector<ReadyQueue> ready(group_resources.size());vector<int> group_busy(group_resources.size());
    vector<vector<int>> resource_groups(nr);
    for(size_t gr=0;gr<group_resources.size();gr++)for(int r:group_resources[gr])resource_groups[r].push_back(gr);
    priority_queue<Event,vector<Event>,greater<Event>> events;
    auto enqueue=[&](uint32_t id){if(state[id]!=1&&state[id]!=3&&state[id]!=4){state[id]=1;ready[node_group[id]].push({-priorities[id],id});}};
    for(uint32_t i=0;i<g.n;i++)if(g.indeg[i]==0){
        if(release_at[g.nodes[i].rank]>0)delayed_roots[g.nodes[i].rank].push_back(i);
        else {enqueue(i);log_event("ready",i,0);}
    }else log_event("dependency",i,0);
    for(uint32_t rank=0;rank<g.ranks;rank++)if(!delayed_roots[rank].empty())events.push({release_at[rank],g.n+rank});
    while(true){
        while(true){
            int selected=-1;pair<double,uint32_t> best;
            for(size_t gr=0;gr<ready.size();gr++)if(group_busy[gr]==0 && !ready[gr].empty())
                if(selected<0 || ready[gr].top()<best){selected=gr;best=ready[gr].top();}
            if(selected<0)break;
            uint32_t id=best.second;ready[selected].pop();const auto& n=g.nodes[id];
            const auto& rs=group_resources[selected];
            for(int r:rs){busy[r]=true;owner[r]=id;for(int gr:resource_groups[r])group_busy[gr]++;}
            log_event("start",id,now);
            state[id]=3;wait_seconds[n.kind]+=now-ready_at[id];
            double duration=n.kind==2?plans[plan_ids[id]].seconds:n.duration;
            if(n.kind==0)cpu_busy[n.rank]+=duration;
            if(n.kind==1)gpu_busy[n.rank]+=duration;
            if(n.kind==2){auto& p=plans[plan_ids[id]];for(int r=0;r<int(g.ranks);r++)if(n.members&(1ULL<<r))comm_busy[r]+=duration;for(size_t l=0;l<p.bytes.size();l++)if(p.bytes[l]>0){link_bytes[l]+=p.bytes[l];link_reserved[l]+=duration;}}
            events.push({now+duration,id});
        }
        if(events.empty())break;
        now=events.top().end;
        do{
            uint32_t id=events.top().id;events.pop();
            if(id>=g.n){for(uint32_t root:delayed_roots.at(id-g.n)){ready_at[root]=now;enqueue(root);log_event("ready",root,now);}continue;}
            const auto& n=g.nodes[id];state[id]=4;log_event("end",id,now);done++;original_done+=n.refs;if(n.kind==2)coll_done++;
            for(int r=0;r<int(g.ranks);r++)if(n.members&(1ULL<<r))rank_done[r]=now;
            for(int r:group_resources[node_group[id]]){busy[r]=false;owner[r]=-1;for(int gr:resource_groups[r])if(--group_busy[gr]<0)throw runtime_error("negative group occupancy");}
            for(uint64_t p=g.offsets[id];p<g.offsets[id+1];p++){uint32_t ch=g.children[p];if(ch>=g.n||g.indeg[ch]==0)throw runtime_error("invalid dependency state");if(--g.indeg[ch]==0){ready_at[ch]=now;log_event("ready",ch,now);enqueue(ch);}}
        }while(!events.empty()&&events.top().end==now);
    }
    struct rusage usage{};getrusage(RUSAGE_SELF,&usage);
    double wall=chrono::duration<double>(chrono::steady_clock::now()-wall_start).count();bool complete=done==g.n&&original_done==g.original;
    ofstream out(argv[3]);out<<setprecision(12)<<"{\n\"complete\":"<<(complete?"true":"false")<<",\n\"nodes_completed\":"<<done<<",\n\"nodes_expected\":"<<g.n<<",\n\"original_nodes_completed\":"<<original_done<<",\n\"original_nodes_expected\":"<<g.original<<",\n\"collectives_completed\":"<<coll_done<<",\n\"makespan_s\":"<<now<<",\n\"wall_s\":"<<wall<<",\n\"peak_rss_kib\":"<<usage.ru_maxrss<<",\n\"model\":\"synchronized collective rounds; atomic endpoint/link reservation; shortest-hop cut-through approximation\",\n";
    out<<"\"alltoall_algorithm\":\""<<net.alltoall<<"\",\n";
    auto array_out=[&](const string& key,const vector<double>& vals){out<<"\""<<key<<"\":[";for(size_t i=0;i<vals.size();i++){if(i)out<<",";out<<vals[i];}out<<"],\n";};
    array_out("rank_completion_s",rank_done);array_out("cpu_busy_s",cpu_busy);array_out("gpu_busy_s",gpu_busy);array_out("collective_endpoint_busy_s",comm_busy);array_out("link_reserved_s",link_reserved);array_out("link_bytes",link_bytes);
    out<<"\"wait_seconds_by_kind\":["<<wait_seconds[0]<<","<<wait_seconds[1]<<","<<wait_seconds[2]<<","<<wait_seconds[3]<<"]\n}\n";
    cout<<"complete="<<complete<<" original="<<original_done<<"/"<<g.original<<" makespan_s="<<now<<" wall_s="<<wall<<"\n";return complete?0:2;
 }catch(const exception& e){cerr<<"orbit_engine: "<<e.what()<<"\n";return 1;}
}
