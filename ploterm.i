/* ploterm.i */
%module ploterm

%include "numpy.i"
%include "typemaps.i"
%include "std_string.i"
%include "std_vector.i"

namespace std {
   %template(FloatVector) vector<float>;
   %template(FloatVectorVector) vector<vector<float> >;
}

%{
  std::string plot(std::vector<float> data, int W, int H);
  std::string heatmap(std::vector<std::vector<float> > data, int W, int H, 
		      std::string color_maps24);
%}

std::string plot(std::vector<float> data, int W, int H);
std::string heatmap(std::vector<std::vector<float> > data, int W, int H, 
		    std::string color_maps24);
