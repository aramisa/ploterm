/* ploterm.i */
%module ploterm

%include "typemaps.i"
%include "std_string.i"
%include "std_vector.i"
namespace std {
   %template(FloatVector) vector<float>;
}

%{
  std::string ascii_plot_simple_wrap(std::vector<float> data, int W, int H);
%}

std::string ascii_plot_simple_wrap(std::vector<float> data, int W, int H);

