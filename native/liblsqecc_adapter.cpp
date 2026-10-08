// Geometry substitution only. Native gate lowering, routing and scheduling remain
// in the unmodified pinned library. CLI rejects unknown flags on stock binaries.
#include <lsqecc/pipelines/slicer.hpp>
#include <lsqecc/layout/dynamic_layouts/edpc_layout.hpp>
#include <lsqecc/layout/dynamic_layouts/compact_layout.hpp>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
namespace { std::string geometry; }
namespace lsqecc {
std::unique_ptr<Layout> original_make_edpc_layout(size_t,size_t,bool,bool,bool,const DistillationOptions&);
std::unique_ptr<Layout> original_make_compact_layout(size_t,const DistillationOptions&,bool);
std::unique_ptr<Layout> overridden(size_t n,const DistillationOptions& d){
 auto l=std::make_unique<LayoutFromSpec>(geometry,d);
 if(l->core_patches().size()!=n)throw std::runtime_error("Geometry must contain exactly the circuit's number of Q cells");
 return l;
}
std::unique_ptr<Layout> make_edpc_layout(size_t n,size_t lanes,bool condensed,bool factories,bool predistilled,const DistillationOptions& d){
 if(!geometry.empty())return overridden(n,d);
 return original_make_edpc_layout(n,lanes,condensed,factories,predistilled,d);
}
std::unique_ptr<Layout> make_compact_layout(size_t n,const DistillationOptions& d,bool no_clogging){
 if(!geometry.empty())return overridden(n,d);
 return original_make_compact_layout(n,d,no_clogging);
}
}
int main(int argc,const char** argv){
 std::vector<const char*> args;
 for(int i=0;i<argc;++i){
  if(std::string(argv[i])=="--geometry-file"){
   if(i+1>=argc){std::cerr<<"Missing geometry file\n";return 2;}
   std::ifstream f(argv[++i]);if(!f){std::cerr<<"Cannot read geometry file\n";return 2;}
   std::ostringstream s;s<<f.rdbuf();geometry=s.str();if(geometry.empty())return 2;
  }else args.push_back(argv[i]);
 }
 return lsqecc::run_slicer_program(args.size(),args.data(),std::cin,std::cout,std::cerr);
}
