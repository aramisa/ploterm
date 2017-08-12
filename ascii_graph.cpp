#include <vector>
#include <iostream>
#include <cmath>
#include <algorithm>

std::vector<float> reduce_data(std::vector<float> data, int W)
{

}

std::vector< std::vector<std::string> > ascii_plot_simple(std::vector<float> data, int W, int H)
{
  // Assuming data comes in clean, and with length W (some prior func
  // has to take care of that). Only allows positive numbers
  std::vector<std::string> CMAP {"▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"};
  int lev_char = CMAP.size();

  // get max and min
  float max_data = *std::max_element(std::begin(data), std::end(data));
  float min_data = *std::min_element(std::begin(data), std::end(data));
  float diff_data = max_data - min_data;
  
  if (max_data == min_data)
    {
      // data is flat
      if (max_data == 0)
	{
	  max_data = 0.1;
	}
      else
	{
	  max_data = max_data + 0.1;
	  min_data = 0;
	}
    }

  std::vector< std::vector<std::string> > C(H, std::vector<std::string>(W, " "));
  
  for (int n=0; n<W; n++)
    {
      // get data normalized in the range 0-1 and scale to the number
      // of rows available.
      float raw_rows = H * ((data[n] - min_data) / diff_data);
      // get the number of full rows
      int full_rows = std::floor(raw_rows);
      // paint the number of full blocks necessary
      for (int x=0; x<full_rows; x++)
	{
	  C[x][n] = CMAP[lev_char - 1];  // full bar
	}
      // check if there is a fraction of a row needed
      float frac_row = raw_rows - full_rows;
      if (frac_row != 0)
	{
	  // scale frac to levels we can afford in row
	  int frac_bar = std::floor(lev_char * frac_row);
	  C[full_rows][n] = CMAP[frac_bar];  // frac bar
	}
    }
  // return
  return C;
}



int main(void)
{
  std::vector<float> data(180, 0);
  std::vector<std::string> CMAP {"_", "▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"};

  int W = 180;
  int H = 10;
  for (int i=0; i<180; i++)
    { 
      data[i] = std::sin(i/6.283185);
    }

  std::vector< std::vector<std::string> > C;
  for (int i=0; i<500000; i++)
    C = ascii_plot_simple(data, W, H);  

  //for(int i=0; i<CMAP.size(); i++){
  //  std::cout<<CMAP[i];
  //}
  for (int j=C.size()-1; j>=0; j--)
    {
      for (int i=0; i<C[j].size(); i++)
	{
	  std::cout<<C[j][i];
	}
      std::cout<<std::endl;
    }
  std::cout<<std::endl;
  return 0;
}
