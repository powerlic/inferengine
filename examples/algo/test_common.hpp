#pragma once
#include <algorithm>
#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <map>
#include <stack>
#include <ctime>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/tracking.hpp>
#include <random>
#include "detector_types.hpp"
#include "detector.hpp"
#include "video_tracker.hpp"

using namespace std;
using namespace std::chrono;

std::stack<high_resolution_clock::time_point> tictoc_stack;

static const string CONFIG_FILE = "/home/licheng/AlgoPro/inferengine/models/faster_rcnn_fpn.json";

void tic() {
  tictoc_stack.push(high_resolution_clock::now());
}

double toc(const string str) {
  std::cout << "Time elapsed: ";
  double diff = duration_cast<milliseconds>( high_resolution_clock::now() - tictoc_stack.top() ).count();
  std::cout<<str<<" "<< diff;
  std::cout << " ms"<<std::endl;
  tictoc_stack.pop();
  return diff;
}

void ReadImgList(string img_list_file, std::vector<string> &img_list)
{
  cout<<img_list_file<<endl;
  std::ifstream infile;
  infile.open(img_list_file);
  std::string line;
  while (std::getline(infile, line))
  {
    img_list.push_back(line);
  }
  infile.close();
}

void split(const string& s, vector<string>& v, const string& c)
{
    string::size_type pos1, pos2;
    pos2 = s.find(c);
    pos1 = 0;
    while(string::npos != pos2)
    {
        v.push_back(s.substr(pos1, pos2-pos1));
         
        pos1 = pos2 + c.size();
        pos2 = s.find(c, pos1);
    }
    if(pos1 != s.length())
        v.push_back(s.substr(pos1));
}

string GetVideoName(string video_path)
{
    vector<string> fileparts, nameparts;
    split(video_path, fileparts, "/");
    split(fileparts[fileparts.size()-1], nameparts, ".");
    return nameparts[0];
}

long CheckBound(cv::Rect &r, int rows, int cols) {
    int x1 = r.x;
    int y1 = r.y;
    int x2 = r.x+r.width-1;
    int y2 = r.y+r.height-1;
    x1 = std::max(0, x1);
    y1 = std::max(0, y1);
    x2 = std::min(x2, cols-1);
    y2 = std::min(y2, rows-1);
    r.x = x1;
    r.y = y1;
    r.width = x2-x1+1;
    r.height = y2-y1+1;
    return 0;
}


std::vector<cv::Scalar> randomColors;

void get_random_colors()
{
    randomColors.resize(256 * 256 * 256);
    for (int i = 0; i < randomColors.size(); ++i)
    {
        randomColors[i] = cv::Scalar(i / 256 / 256, (i / 256) % 256, i % 256);
    }
    for (int i = randomColors.size(); i > 0; --i)
    {
        int p = std::rand() % i;
        auto tmp = randomColors[i];
        randomColors[i] = randomColors[p];
        randomColors[p] = tmp;
    }
}

