/* ploterm.i */
%module ploterm

%include "typemaps.i"
%include "std_string.i"
%include "std_vector.i"
namespace std {
   %template(FloatVector) vector<float>;
}

%{
  #include "ploterm.h"
%}

%include "ploterm.h"
