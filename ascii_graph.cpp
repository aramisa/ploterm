#include "ploterm.h"


void get_min_max(std::vector<float> &data_short, float &max_data, float &min_data,
		 float &diff_data)
{
  // get max and min
  max_data = *std::max_element(std::begin(data_short), std::end(data_short));
  min_data = *std::min_element(std::begin(data_short), std::end(data_short));
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
  diff_data = max_data - min_data;
}


std::vector<float> reduce_data(std::vector<float> &data, int W)
{
  std::vector<float> data_out(W, 0);
  if (data.size() < W)
    {
      // resampling data
      float np_inc = (data.size() - 1) / (float)W;
      float realp = np_inc;
      data_out[0] = data[0];
      for (int i=1; i<W-1; i++)
	{
	  int p1 = std::floor(realp);
	  int p2 = p1 + 1;
	  float f1 = p2 - realp;
	  float f2 = realp - p1;
	  data_out[i] = data[p1] * f1 + data[p2] * f2;
	  realp += np_inc;
	}
      data_out[W-1] = data.back();
      return data_out;
    }
  else if(data.size() == W){
    return data;
  }
  // simple approach, split vector in W segments and average points in each segment
  int step = std::ceil(data.size() / (float)W);
  std::vector<int> ranges(W+1, step);
  ranges[0] = 0;
  // now is overcomplete, discard until sizes match
  int idx = 1;
  while (std::accumulate(ranges.begin(), ranges.end(), 0) > data.size())
    {
      --ranges[idx++];
    }

  auto start = data.begin();
  auto end = data.begin();

  for (int i=0; i<W; i++)
    {
      start = std::next(start, ranges[i]);
      end = std::next(end, ranges[i+1]);
      data_out[i] = std::accumulate(start, end, 0.0) / (float)ranges[i+1];
    }
  // should check for NaNs
  return data_out;
}


std::vector<std::string> make_y_axis(std::vector<float> &data, std::vector<float> &data_short,
				     float& max_data, float& min_data, float &diff_data,
				     int& W, int H)
{
  int Yaxis_size = 5;
  bool Yaxis_set = false;
  std::vector<std::string> Yaxis(H, "");

  if (W < 15)
    {
      // Not enough space to plot Yaxis
      data_short = reduce_data(data, W - 2);
      get_min_max(data_short, max_data, min_data, diff_data);
      return std::vector<std::string>(H, " \x1B[1;33m|\x1B[0m");
    }
  while (Yaxis_set == false)
    {
      // resize data to size W-5 (best guess), to estimate Y-ticks
      data_short = reduce_data(data, W - Yaxis_size);
      get_min_max(data_short, max_data, min_data, diff_data);
      int max_size = 0;
      
      // draw Y axis now: first column of C
      for (int x=0; x<H; x++)
	{
	  std::stringstream yaxis_ss;
	  if (x == 0)
	    {
	      yaxis_ss << std::setprecision(2) << min_data;
	    }
	  else
	    {
	      if (x == H-1)
		{
		  yaxis_ss << std::setprecision(2) << max_data;
		}
	      else
		{
		  if (x % 2 == 0)
		    {
		      yaxis_ss << std::setprecision(2) << (min_data + (x + 1) * diff_data / H);
		    }
		}
	    }
	  std::stringstream yaxis_ss2;
	  while (yaxis_ss.str().size() + yaxis_ss2.str().size() < Yaxis_size - 1)
	    {
	      yaxis_ss2 << " ";
	    }
	  yaxis_ss2 << yaxis_ss.str();
	  yaxis_ss2 << "|";
	  Yaxis[x] = yaxis_ss2.str();
	  if (Yaxis[x].size() > max_size)
	    {
	      max_size = Yaxis[x].size();
	    }
	  Yaxis[x] = "\x1B[1;33m" + Yaxis[x] + "\x1B[0m";
	}
      if(max_size <= Yaxis_size)
	{
	  Yaxis_set = true;
	  W = data_short.size() + max_size; //real_W
	}
      else
	{
	  Yaxis_size = max_size;
	}
    }
  return Yaxis;
}


std::vector<std::string> make_x_axis(int plot_W, int real_W, int max_X)
{
  std::vector<std::string> Xaxis(plot_W + 1, " ");
  std::stringstream pre_xaxis;
  std::cout<<real_W<<" "<<plot_W<<std::endl;

  if (real_W < 15)
    {
      // refuse to plot axis 
      Xaxis = std::vector<std::string>(plot_W + 1, "\x1B[0;33m¯\x1B[0m");
      Xaxis[0] = "\x1B[0;33m¯|\x1B[0m";
      return Xaxis;
    }
  for (int i=0; i<real_W - plot_W - 2; i++)
    {
      pre_xaxis << "¯";
    }
  pre_xaxis << "Ō|";
  Xaxis[0] = std::string("\x1B[0;33m") + pre_xaxis.str() + std::string("\x1B[0m");

  std::stringstream buffer;
  buffer << max_X;
  int ticksize = buffer.str().size() + 3;  // one dec position + bar
  int nticks = std::floor(plot_W / (float)ticksize);
  int lnleft = plot_W - nticks * ticksize;
  std::stringstream xticks;
  for (int i=0; i<nticks; i++)
    {
      float tickval = (i + 1) / (float)nticks * max_X;
      std::stringstream tickval_ss;
      tickval_ss << std::setprecision(1) << std::fixed << tickval;
      for(int j=0; j<ticksize - tickval_ss.str().size() - 1; j++)
	{
	  xticks << " ";
	}
      if (lnleft)
	{
	  xticks << " ";
	  lnleft--;
	}
      xticks << tickval_ss.str();
      xticks << "|";
    }
  std::string xticks_str = xticks.str();
  for (int i=1; i<=plot_W; i++)
    {
      Xaxis[i] = std::string("\x1B[0;33m") + xticks_str[i-1] + std::string("\x1B[0m");
    }
  return Xaxis;
}


std::vector< std::vector<std::string> > ascii_plot_simple(std::vector<float> &data, int W, int H)
{
  std::vector<std::string> CMAP {"▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"};
  const size_t lev_char = CMAP.size();

  std::vector<float> data_short;
  float max_data;
  float min_data;
  float diff_data;
  int real_W = W;
  int plot_H = H - 1; //x axis
  // abort conditions
  if ((W < 3) || (data.size() < 2))
    {
      // refuse to plot
      std::vector< std::vector<std::string> > C(1, std::vector<std::string>(1, "X"));
      return C;
    }

  std::vector<std::string> Yaxis = make_y_axis(data, data_short, max_data, min_data, diff_data, real_W, plot_H);
  std::vector<std::string> Xaxis = make_x_axis(data_short.size(), real_W, data.size());
  std::vector< std::vector<std::string> > C(H, std::vector<std::string>(data_short.size() + 1, " "));

  for (int x=1; x<H; x++)  // skip first for Xaxis
    {
      C[x][0] = Yaxis[x-1];
    }
  for (int n=0; n<data_short.size() + 1; n++)  // skip first for Xaxis
    {
      C[0][n] = Xaxis[n];
    }
  for (int n=0; n<data_short.size(); n++)
    {
      // get data normalized in the range 0-1 and scale to the number
      // of rows available.
      float raw_rows = plot_H * ((data_short[n] - min_data) / diff_data);
      // get the number of full rows
      int full_rows = std::floor(raw_rows);
      // paint the number of full blocks necessary
      for (int x=0; x<full_rows; x++)
      {
      	  C[x + 1][n + 1] = CMAP[lev_char - 1];  // full bar
      	}
      // check if there is a fraction of a row needed
      float frac_row = raw_rows - full_rows;
      if (frac_row != 0)
	{
	  // scale frac to levels we can afford in row
	  int frac_bar = std::floor(lev_char * frac_row);
	  C[full_rows + 1][n + 1] = CMAP[frac_bar];  // frac bar
	}
    }
  // return
  return C;
}

std::string ascii_plot_simple_wrap(std::vector<float> data, int W, int H)
{
  std::vector< std::vector<std::string> > C;
  std::string out = "";
  C = ascii_plot_simple(data, W, H);
  for (int j=C.size()-1; j>=0; j--)
    {
      for (int i=0; i<C[j].size(); i++)
  	{
  	  out += C[j][i];
  	}
      out += "\n";
    }
  out += "\n";
  return out;
}

int main(void)
{
  std::vector<float> data(100, 0);
  std::vector<std::string> CMAP {"_", "▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"};

  int W = 150;
  int H = 10;
  for (int i=0; i<100; i++)
    { 
      data[i] = std::cos(i/6.283185);
    }

  std::vector< std::vector<std::string> > C;
  for (int i=0; i<1; i++)
    C = ascii_plot_simple(data, W, H);  
  for (int j=C.size()-1; j>=0; j--)
    {
      for (int i=0; i<C[j].size(); i++)
  	{
  	  std::cout<<C[j][i];
  	}
      std::cout<<std::endl;
    }
  std::cout<<std::endl;


  //std::stringstream os;
  //ascii_plot_stringstream(os, data, W, H);
  //std::cout<<os<<std::endl;
  return 0;
}
