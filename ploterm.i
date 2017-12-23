/* ploterm.i */
%module ploterm

%include "stdint.i"
%include "numpy.i"
%include "typemaps.i"
%include "std_string.i"
%include "std_vector.i"

namespace std {
   %template(FloatVector) vector<float>;
   %template(FloatVectorVector) vector<vector<float> >;
   %template(Uint8VectorVectorVector) vector<vector<vector<uint8_t> > >;
}

%{
  std::string plot(std::vector<float> data, int W, int H);
  std::string heatmap(std::vector<std::vector<float> > data, int W, int H, 
		      std::string color_maps24);
  std::string image(std::vector<std::vector<std::vector<uint8_t> > > img, int W, int H)
%}

std::string plot(std::vector<float> data, int W, int H);
std::string heatmap(std::vector<std::vector<float> > data, int W, int H, 
		    std::string color_maps24);
std::string image(std::vector<std::vector<std::vector<uint8_t> > > img, int W, int H);
