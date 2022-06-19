#include "FormConversion.hh"
#include <iostream>
#include <queue>

// convert 1-form defined on halfedges to 0-form on vertices
// xi -> phi

template <typename Scalar>
void OverlayProblem::form_conversion(Mesh<Scalar> & m, const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& xi, Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& phi){

  int n_h = m.n_halfedges();
  int n_f = m.n_faces();
  int n_v = m.n_vertices();
  phi.setZero(n_h);
  std::vector<int> start_hs;
  std::vector<bool> done(n_h, false);
  std::vector<bool> visited(n_f, false);
  std::vector<bool> visited2(n_f, false);
  std::vector<bool> assigned(n_h, false);

  // check the closeness condition on \xi
  // for(int i = 0; i < n_f; i++){
  //   int hi = m.h[i];
  //   int hj = m.n[hi];
  //   int hk = m.n[hj];
  //   auto sum = xi[hi] + xi[hj] + xi[hk];
  //   if(std::abs(sum) > 1e-14){
  //     std::cout<<"closeness broken on face "<<i<<" with sum = "<<sum<<std::endl;
  //   }
  // }

  int h = 0;
  int i_start = 0;

  while(true){ // repeat to handle multiple disk components
    if(!start_hs.empty()){
      if(i_start >= int(start_hs.size())) break; // all start_hs processed
      h = start_hs[i_start];
      if(done[m.opp[h]] || done[h]) std::cerr << "\n\n\n\n\nERROR: multi start_h in same component!\n\n\n\n\n";
      if(m.f[h] < 0) std::cerr << "\n\n\n\n\nERROR: start_h has no face!\n\n\n\n\n";
      h = m.n[m.n[h]];
      i_start++;
    }else{
      //use boundary halfedge if exists
      for(int i = 0; i < n_h; ++i)
      {
        if(m.f[i] < 0 && !done[i])
        {
          h = m.n[m.n[m.opp[i]]];
          done[i] = true;
          break;
        }
      }
    }
    int to_v = h == -1 ? -1: m.to[h];
    // std::cerr << " h="<<h << "("<<m.to[m.opp(h)]<<") ";
    if(h < 0) break;
    phi[h] = 0.0;
    assigned[h] = true;
    h = m.n[h];
    phi[h] = xi[h];
    phi[m.opp[h]] = 0;
    assigned[h] = true;
    assigned[m.opp[h]] = true;
    
    // layout the rest of the mesh by BFS
    std::queue<int> q;
    q.push(h);
    visited[m.f[h]] = true;
    
    while(!q.empty())
    {
      h = q.front();
      q.pop();
      
      int hn = m.n[h];
      int hp = m.n[hn];
      
      phi[hn] = phi[h] + xi[hn];
      assigned[hn] = true;
      
      int hno = m.opp[hn];
      int hpo = m.opp[hp];
      done[hno] = true;
      done[hpo] = true;
      if(m.f[hno] >= 0 && !visited[m.f[hno]]){
        visited[m.f[hno]] = true;
        phi[hno] = phi[h];
        phi[m.n[m.n[hno]]] = phi[hn];
        assigned[hno] = true;
        assigned[m.n[m.n[hno]]] = true;
        q.push(hno);
      }else if(m.f[hno] == -1){ // handle boundary halfedge
        phi[hno] = phi[h];
        assigned[hno] = true;
      }

      if(m.f[hpo] >= 0 && !visited[m.f[hpo]]){
        visited[m.f[hpo]] = true;
        phi[hpo] = phi[hn];
        phi[m.n[m.n[hpo]]] = phi[hp];
        assigned[hpo] = true;
        assigned[m.n[m.n[hpo]]] = true;
        q.push(hpo);
      }else if(m.f[hpo] == -1){
        phi[hpo] = phi[hn];
        assigned[hpo] = true;
      }
    }

    // just mark entire component's h's as done (to not have the next iteration pick up an unreached part of the same compo)
    q.push(h);
    visited2[m.f[h]] = true;
    while(!q.empty())
    {
      h = q.front();
      q.pop();
      
      int hn = m.n[h];
      int hp = m.n[hn];

      int hno = m.opp[hn];
      int hpo = m.opp[hp];
      done[hno] = true;
      done[hpo] = true;
      if(m.f[hno] >= 0 && !visited2[m.f[hno]]){
        visited2[m.f[hno]] = true;
        q.push(hno);
      }
      if(m.f[hpo] >= 0 && !visited2[m.f[hpo]]){
        visited2[m.f[hpo]] = true;
        q.push(hpo);
      }
    }
    
    h = -1;
  }

  // convert phi on halfedges to vertices
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> phi_v; phi_v.setZero(n_v);
  std::vector<std::vector<Scalar>> valts(n_v, std::vector<Scalar>());
  for(int i=0;i<n_h;i++){
    phi_v[m.to[i]] = phi[i];
    valts[m.to[i]].push_back(phi[i]);
  }
  phi = phi_v;
  for(int i=0;i<valts.size();i++){
    auto values = valts[i];
    Scalar avg = 0.0;
    for(int j = 0;j<values.size();j++){
      avg += values[j];
    }
    avg /= values.size();
    for(int j = 0;j<values.size();j++){
      Scalar diff = abs(values[j]-avg);
      if(diff > 1e-12){
        std::cout<<"phi mismatch on vertex "<<i<<"["<<values[j]<<","<<avg<<"] - "<<"(diff = "<<diff<<")"<<std::endl;
      }
    }
    phi[i] = avg;
  }

}

template void OverlayProblem::form_conversion<double>(Mesh<double> & m, const Eigen::Matrix<double, Eigen::Dynamic, 1>& xi, Eigen::Matrix<double, Eigen::Dynamic, 1>& phi);
template void OverlayProblem::form_conversion<mpfr::mpreal>(Mesh<mpfr::mpreal> & m, const Eigen::Matrix<mpfr::mpreal, Eigen::Dynamic, 1>& xi, Eigen::Matrix<mpfr::mpreal, Eigen::Dynamic, 1>& phi);
