// Calls the pinned native generators; does not reimplement their geometry.
#include <lsqecc/layout/dynamic_layouts/edpc_layout.hpp>
#include <lsqecc/layout/dynamic_layouts/compact_layout.hpp>
#include <iostream>
#include <string>
#include <vector>
int main(int argc,char**argv){
 if(argc!=3){std::cerr<<"usage: export_layout compact|compact_no_clogging|edpc qubits\n";return 2;}
 std::string mode=argv[1];size_t n=std::stoul(argv[2]);if(n<1)return 2;
 lsqecc::DistillationOptions options;std::unique_ptr<lsqecc::Layout> layout;
 if(mode=="edpc")layout=lsqecc::make_edpc_layout(n,1,false,false,false,options);
 else if(mode=="compact"||mode=="compact_no_clogging")layout=lsqecc::make_compact_layout(n,options,mode=="compact_no_clogging");
 else return 2;
 auto end=layout->furthest_cell();std::vector<std::string> grid(end.row+1,std::string(end.col+1,'r'));
 auto put=[&](lsqecc::Cell c,char x){if(grid[c.row][c.col]!='r')throw std::runtime_error("Overlapping resource");grid[c.row][c.col]=x;};
 for(auto& p:layout->core_patches())for(auto c:p.get_cells())put(c,'Q');
 for(auto c:layout->ancilla_location())put(c,'A');for(auto c:layout->dead_location())put(c,'X');
 for(auto c:layout->reserved_for_magic_states())put(c,'M');for(auto c:layout->predistilled_y_states())put(c,'Y');
 // These default generators use a single explicit factory (compact) or M cells.
 if(layout->distillation_regions().size()>10)throw std::runtime_error("Too many factories for unique ASCII labels");
 for(size_t i=0;i<layout->distillation_regions().size();++i)for(auto c:layout->distillation_regions()[i].sub_cells)put(c.cell,char('0'+i));
 for(auto& row:grid)std::cout<<row<<'\n';
}
