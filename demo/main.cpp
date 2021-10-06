/*
DeformXfer (https://github.com/jerenchen/deformxfer) - demo program

An Eigen/libIGL (C++) implementation of "Deformation transfer for triangle meshes"
  (https://people.csail.mit.edu/sumner/research/deftransfer/), largely based on
  the Python implementation of "Landmark-guided deformation transfer of template
  facial expressions for automatic generation of avatar blendshapes"
  (https://github.com/diegothomas/Avatar-generation-3DRW2019-)

This source code is licensed under the BSD 3-Clause "New" or "Revised" License
*/
#include <deformxfer.h>

#include <iostream>
#include <chrono>

#include <igl/combine.h>
#include <igl/per_vertex_normals.h>
#include <igl/read_triangle_mesh.h>
#include <igl/opengl/glfw/Viewer.h>


const double pi = std::acos(-1);
double anim_t = 0.0;
int shape_curr = 24;
int shape_prev = 0;
bool show_morph = false;

// util func to read example landmark data
std::vector<int> read_landmarks(const std::string& filename)
{
  std::cout << "Reading landmarks: " << filename << "..." << std::endl;
  std::vector<int> markers;
  std::ifstream fin;
  fin.open(filename, std::ifstream::in);
  int s_;
  while(fin >> s_)
  {
      markers.push_back(s_);
  }
  fin.close();
  return markers;
}

int main(int argc, char *argv[])
{
  Eigen::MatrixXd V, SV, TV, DV, RV;
  Eigen::MatrixXi F, SF, TF;

  // variables to store morphed targets & correspondences
  std::vector<Eigen::MatrixXd> X;
  std::vector<std::pair<int, int> > M;
  // variables to store constaint indices & conditions
  Eigen::VectorXi CI;
  Eigen::MatrixXd CC;

  // begin to read inputs

  std::vector<Eigen::MatrixXd> BS(48);
  for (unsigned int ii = 0; ii < BS.size(); ++ii)
  {
    std::string filename(BLENDSHAPE_DIR);
    std::string no("0" + std::to_string(ii));
    filename += no.substr(no.size() - 2, 2) + "m.obj";
    std::cout << "Reading " << filename << " ..." << std::endl;
    igl::read_triangle_mesh(filename, BS[ii], SF);
  }
  igl::read_triangle_mesh(std::string(BLENDSHAPE_DIR) + "Neutralm.obj", SV, SF);
  igl::read_triangle_mesh(std::string(INPUTSHAPE_DIR) + "Horse.ply", TV, TF);

  DV = SV;
  RV = TV;

  std::string landmark_dir(LANDMARKS_DIR);
  auto landmarks = read_landmarks(landmark_dir + "SOURCE/FACIAL_LANDMARKS.txt");
  auto horse_markers = read_landmarks(landmark_dir + "landmarks_Horse.txt");

  // rescale target to match source
  auto SB = SV.colwise().maxCoeff() - SV.colwise().minCoeff();
  auto TB = TV.colwise().maxCoeff() - TV.colwise().minCoeff();
  auto rescale = SB.sum() / TB.sum();
  auto center = (TV.colwise().maxCoeff() + TV.colwise().minCoeff()) / 2.0;
  TV = rescale * (TV.rowwise() - center);

  // offest from target to source
  auto offset = SV.row(landmarks[13]) - TV.row(horse_markers[13]);
  TV = TV.rowwise() + offset;

  // adding marker pairs for mapping mesh correspondences
  std::vector<std::pair<int, int> > markers;
  for (unsigned int ii = 0; ii < landmarks.size(); ++ii)
  {
    markers.push_back(std::pair<int, int>(landmarks[ii], horse_markers[ii]));
  }

  // adding backward landmarks as positional constraints
  auto backward_markers = read_landmarks(landmark_dir + "SOURCE/BACKWARD_LANDMARKS.txt");
  CI.resize(backward_markers.size());
  CC.resize(backward_markers.size(), 3);
  int ci = 0;
  for (auto idx : backward_markers)
  {
    int index;
    auto D = (TV.rowwise() - SV.row(idx)).cwiseAbs2().rowwise().sum();
    D.minCoeff(&index);
    CI(ci) = index;
    CC.row(ci) = TV.row(index);
    ++ci;
  }

  // create dx object
  dx::DeformXfer dxo;
  dxo.keep_progress = true;
  dxo.Wc = {0.1, 1.0, 10.0};
  dxo.Ws = 0.01;
  dxo.Wi = 0;

  auto start = std::chrono::high_resolution_clock::now();
  dxo.initialize(TV, TF);
  auto elapsed = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start);
  std::cout << "Init time: " << elapsed.count() << "s\n";

  // map correspondences
  start = std::chrono::high_resolution_clock::now();
  dxo.map_correspondences(SV, SF, TV, TF, markers, X, M);
  elapsed = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start);
  std::cout << "Map correspondences time: " << elapsed.count() << "s\n";

  // precompute large AtA matrices with the indices of contraint conditions
  start = std::chrono::high_resolution_clock::now();
  dxo.precompute(CI);
  elapsed = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start);
  std::cout << "Pre-compute time: " << elapsed.count() << "s\n";

  // set up viewer & callbacks
  igl::opengl::glfw::Viewer viewer;

  const auto& key_down =
    [&](igl::opengl::glfw::Viewer& view, unsigned char key, int mod)->bool
  {
    switch(key)
    {
      case ' ':
      {
        view.core().is_animating ^= 1;
        anim_t = 0;
        break;
      }
      case 'M':
      case 'm':
      {
        show_morph ^= 1;
        DV = SV;
        anim_t = 0;
        break;
      }
      default:
        return false;
    }
    return true;
  };

  const auto& pre_draw = [&](igl::opengl::glfw::Viewer& view)->bool
  {
    if(viewer.core().is_animating)
    {
      if (anim_t >= 1.0)
      {
        shape_prev = shape_curr;
        shape_curr = std::rand() % BS.size();
        anim_t = 0.0;
      }

      if (show_morph)
      {
        int idx = std::floor(anim_t * (X.size() + 1) - 1);
        RV = idx < 0? TV : X[idx];
        anim_t += 0.01;
      }
      else
      {
        // smoothstep interpolation
        double w = anim_t * anim_t * (3 - 2 * anim_t);
        DV = (1 - w) * BS[shape_prev] + w * BS[shape_curr];
        dxo.transfer(SV, SF, DV, CC, M, RV);
        anim_t += 0.05;
      }
    }

    std::vector<Eigen::MatrixXd> VV;
    std::vector<Eigen::MatrixXi> FF;
    VV.push_back(DV.rowwise() + Eigen::RowVector3d(-100, 0, 0));
    FF.push_back(SF);
    VV.push_back(RV.rowwise() + Eigen::RowVector3d(100, 0, 0));
    FF.push_back(TF);
    igl::combine(VV, FF, V, F);
    view.data().set_mesh(V, F);
    // draw the constraint markers
    Eigen::MatrixXd N;
    igl::per_vertex_normals(VV[0], SF, N);
    viewer.data().set_points((VV[0] + N)(landmarks, Eigen::all), Eigen::RowVector3d(0,1,0));
    igl::per_vertex_normals(VV[1], TF, N);
    viewer.data().add_points((VV[1] + N)(horse_markers, Eigen::all), Eigen::RowVector3d(1,0,0));
    return false;
  };
  pre_draw(viewer); // this will frame meshes in view
  
  viewer.callback_pre_draw = pre_draw;
  viewer.callback_key_down = key_down;
  viewer.core().is_animating = true;
  viewer.data().point_size = 6;

  std::cout << "Press [m] to toggle the 'morph topology' mode." << std::endl;
  return viewer.launch(true, false, "DeformXfer");
}
