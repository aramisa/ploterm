# PLOTERM

## Synopsis

Ploterm is a minimalistic plotting library for command-line
applications, with C++ and Python bindings.

![██▇▄▁ Loading image ▁▃▅██](misc/ploterm.jpg "Ploterm in action.")

The library is very new, so expect many changes.

## Code Example

```
#include <vector>
#include "ploterm.h"

int main(void)
{
    std::vector<float> data(100);
    int width = 50;
    int height = 10;

    for (int i=0; i<100; i++)
    {
        data[i] = std::cos(i/6.283185);
    }

    std::string plt = plot(data, width, height);

    std::cout<<plt<<std::endl;

    return 0;
}
```

## Motivation

The purpose of this library is to provide a very simple and readily
available way to visualize graphically an arbitrarily long array of
numbers. For example a probability distribution, or the temporal
evolution of a variable.

Ploterm does not require any type of graphical user interface, or
opening ports and setting up a web server to visualize the information
from a remote machine in the browser: it works right at the terminal,
where you run the commands. On top of that, it has a very small
footprint, and no dependencies besides C++ 11; and Swig, if Python
bindings are necessary.

In the future it will be extended to allow more than one variable
plotted, and to also visualize 2D information, like images or heat
maps.

## Installation

Minimal build and installation functionality with Python 2.7 bindings
is provided through CMake. However, the library is small and simple
enough to be directly included in any project that requires it.

## API Reference

At present, the only exposed method to interact with the library is
the ```plot``` function:

```
std::string plot(std::vector<float> data, int W, int H);
```

This function returns a string, formatted to have width W and height
H, with the plotted data. The actual size of the string may be larger
than H * W due to non-printable characters.

If the data array to plot is longer or shorter than W (minus the
axis), a new vector of the right size will be interpolated and
plotted.

## Tests

## Contributors

Arnau Ramisa (arnauramisa@gmail.com)

## License

This software is under the BSD License 2.0