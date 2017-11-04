/* ploterm.i */
%module ploterm

%include "numpy.i"
%include "typemaps.i"
%include "std_string.i"
%include "std_vector.i"

namespace std {
   %template(FloatVector) vector<float>;
}

%{
  std::string plot(std::vector<float> data, int W, int H);
%}

std::string plot(std::vector<float> data, int W, int H);

