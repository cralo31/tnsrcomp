#include <iostream>
#include <RcppArmadillo.h>
#include <vector>
// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;
using namespace std;
using namespace arma;

/* ####### Script for UO-NTF ####### */

/* Kronecker's Product */
// [[Rcpp::export]]
arma::mat kro_prod(arma::mat A, arma::mat B) {
  
  // Multiply each entry in A to the entire matrix of B, then join_rows and join_cols to bind 
  // final product
  
  const int rowA = A.n_rows, rowB = B.n_rows, colA = A.n_cols, colB = B.n_cols;
  mat AB(rowA * rowB, colA * colB, fill::zeros);
  int startRow = 0, startCol = 0;
  
  // i loops till rowA
  for (int i = 0; i < rowA; i++) {
    
    // j loops till colA
    for (int j = 0; j < colA; j++) {
      startRow = i * rowB; 
      startCol = j * colB;
      
      // k loops till rowB
      for (int k = 0; k < rowB; k++) {
        
        // l loops till colB
        for (int l = 0; l < colB; l++) {
          AB(startRow + k, startCol + l) = A(i,j) * B(k,l);
        } 
      }
    }
  }
  return AB;
}


/* Khatri-Rao Product */
// [[Rcpp::export]]
arma::mat krao_prod(arma::mat A, arma::mat B) {
 
 int I = A.n_rows;
 int J = B.n_rows;
 int count = 0;
 mat res(I*J, A.n_cols, fill::zeros);
 
 for (int i = 0; i < I; i++) {
   for (int j = 0; j < J; j++) {
     res.row(count) = A.row(i) % B.row(j);
     count++;
   }
 }
 return (res);
}
 
/* Unfold a order-3 tensor and returns a list
Consider generalization to the n-th order in the future */
// [[Rcpp::export]]
List unfold_ten(arma::cube tens) {
  
  int I = tens.n_rows, J = tens.n_cols, K = tens.n_slices;
  mat mode0(I, J * K, fill::zeros);
  mat mode1(J, I * K, fill::zeros);
  mat mode2(K, I * J, fill::zeros);
  
  // Unfold along the slices
  for (int k = 0; k < K; k++) {
    mode0.cols(J * k, J * k + (J - 1)) = tens.slice(k);
    mode1.cols(I * k, I * k + (I - 1)) = tens.slice(k).t();
    mode2.row(k) = vectorise(tens.slice(k), 0).t();
  }
  
  // Return the unfoldings
  return List::create(_["mode1"] = mode0, _["mode2"] = mode1, _["mode3"] = mode2);
}

/* Fold a mode-n unfolding back into a tensor */
// [[Rcpp::export]]
arma::cube fold_ten(arma::mat A, arma::vec dim, const int mode) {
  
  cube res(dim(0), dim(1), dim(2), fill::zeros);
  mat curr;
  
  if (mode == 1) {
    // Mode-1 folding
    for (int k = 0; k < dim(2); k++) {
      res.slice(k) = A.cols(dim(1)*k, dim(1)*k+(dim(1)-1));
    }
    
  } else if (mode == 2) {
    // Mode-2 folding
    for (int k = 0; k < dim(2); k++) {
      res.slice(k) = A.cols(dim(0)*k, dim(0)*k+(dim(0)-1)).t();
    }
    
  } else if (mode == 3){
    // Mode-3 folding
    for (int k = 0; k < dim(2); k++) {
      curr = A.row(k);
      curr.reshape(dim(0), dim(1));
      res.slice(k) = curr;
    }
  }
  return res;
}


// Computes the frobenius norm of the objective function
// [[Rcpp::export]]
float f_norm(arma::cube tens, arma::cube approx) {
  
  float sum = sqrt(accu(square(tens - approx)));
  return(sum);
}

// The tensor folding part can be simplified, should come back to this

// Compute the product between a tensor and a matrix
// [[Rcpp::export]]
arma::cube tens_mat(arma::cube tens, arma::mat X, int mode) {
  
  // Extract dimension info from input tensor
  double trow = tens.n_rows, tcol = tens.n_cols, tslc = tens.n_slices;
  vec dim_new = { trow, tcol, tslc };
  
  // First check if the dimensions of the input tensor
  // and matrix to determine if the operation is valid
  if (X.n_cols != dim_new(mode - 1)) {
    cout << "Dimension for multiplication does not match." << endl;
    exit (0);
  }
  
  mat mat_mode = unfold_ten(tens)[mode - 1];
  mat prod = X * mat_mode;
  dim_new(mode-1) = prod.n_rows;
  
  // Fold new product matrix back to tensor
  cube f_tens = fold_ten(prod, dim_new, mode);
  
  return(f_tens);
}

/*-----------------------------------------------------*/
/* Core to Full Tensor and Vice Versa */

// Reconstruct the full tensor using tensor core and factor matrices
// [[Rcpp::export]]
arma::cube core_ten(List core_mat) {
  
  cube res = tens_mat(core_mat[0], core_mat[1], 1);
  for (int i = 2; i <= 3; i++) {
    res = tens_mat(res, core_mat[i], i);
  }
  return(res);  
}

// Find reduced core given the full tensor and factor matrices
// [[Rcpp::export]]
arma::cube find_core(arma::cube tens, List mats) {
  
  mat X1 = mats[1], X2 = mats[2], X3 = mats[3];
  cube G = tens_mat(tens, X1.t(), 1);
  G = tens_mat(G, X2.t(), 2);
  G = tens_mat(G, X3.t(), 3);
  
  return(G);
} 

/*-----------------------------------------------------*/


// Computes a tensor as the sum of rank-one tensors (product of vectors)
// Needs to initialize memory for new tensor or else numbers will be extremely funky
// [[Rcpp::export]]
arma::cube vec_tensor (arma::mat A, arma::mat B, arma::mat C) {
  
  vec colC;
  mat ab_prod;
  cube temp(A.n_rows, B.n_rows, C.n_rows, fill::zeros);
  cube tens_sum(A.n_rows, B.n_rows, C.n_rows, fill::zeros);
  
  int K = A.n_cols;
  
  for (int r = 0; r < K; r++) {
    ab_prod = A.col(r) * B.col(r).t();
    colC = C.col(r);
    for (int j = 0, ele = colC.n_elem; j < ele; j++) {
      temp.slice(j) = ab_prod * colC(j);
    }
    tens_sum += temp;
  }
  return tens_sum;
}

/*---------------------------------------------------*/
/* Algorithm for Bi-orthogonal CP Decomposition */

/* ALS update for non-negative factor matrix */
// [[Rcpp::export]]
List als_up (List X_list, List tens_list, int mode, bool nng) {
  
  // Calculate the kronecker product between the factor matrices and the core tensor
  mat X = X_list[mode-1], kp;
  if (mode == 1) {
    kp = krao_prod(tens_list[2], tens_list[1]);
  } else if (mode == 2) {
    kp = krao_prod(tens_list[2], tens_list[0]);
  } else if (mode == 3) {
    kp = krao_prod(tens_list[1], tens_list[0]);
  }
  
  // Update A w/ ALS update
  mat A = X * kp * pinv(kp.t() * kp);
  if (nng) {
    A.elem(find(A < 0)).zeros();  
  }
  tens_list[mode - 1] = A;
  
  return(tens_list);
}

/* Wen and Yin's Method */
// [[Rcpp::export]]
List wy_bls (List X_modes, List tens_list, int mode, float tau, float beta) {
  
  // Read in the input mode-n matrices
  mat X_n = X_modes[mode-1], A = tens_list[mode-1], krao_t;
  float left, right_1, right_2, right, k = A.n_cols, count = 0;
  mat I_2 = eye(2*k, 2*k), diff_mat, grad_G;
  
  // Compute the gradient of the loss function wrt A
  if (mode == 1) {
    krao_t = krao_prod(tens_list[2], tens_list[1]);
  } else if (mode == 2) {
    krao_t = krao_prod(tens_list[2], tens_list[0]);
  } else if (mode == 3) {
    krao_t = krao_prod(tens_list[1], tens_list[0]);
  }
  mat A_grad = (A * trans(krao_t) - X_n) * krao_t;
  
  // Find suitable step size via backtracking line search
  // by checking Wolfe's condition
  diff_mat = X_n - A * trans(krao_t);
  grad_G = A_grad * trans(krao_t);
  right_1 = accu(square(diff_mat));
  right_2 = accu(square(A_grad)); 
  do {
    left = accu(square(diff_mat + tau * grad_G));
    right = right_1 - right_2 * tau/2;   
    tau = tau * beta;
    count++;
  } while (left > right);
  tau = tau / beta; // compensate for dividing it by one extra time in the do while loop
  
  // Use the Woodbury Formula 
  mat U = join_rows(A_grad, A);
  mat V = join_rows(A, -A_grad);
  
  // Orthogonal Preserving Update
  mat A_new = A - tau * U * inv(I_2 + tau/2 * V.t() * U) * V.t() * A;
  tens_list[mode-1] = A_new;
  tens_list.push_back(tau);
  
  return (tens_list);
} 

// Algorithm for Bi-orthogonal CP decomposition 
// [[Rcpp::export]]
List bontf (arma::cube X, List X_n, List tnsr_list, int rank, int iter, double tol, bool nng) {
  
  /* Initialization of parameters */
  mat resmat;                             // Store factorization result
  int count = 0;                          // Keep track of count
  vec all_res(iter+1), all_tol(iter+1); 
  List A1, A2, A3;
  double tau = 1, beta = 0.5;
  
  all_res(count) = all_tol(count) = f_norm(X, vec_tensor(tnsr_list[0], tnsr_list[1], tnsr_list[2]));
  
  /* Begin the update */
  while ((count != iter) && (all_res(count) > tol)) {
    
    // Shrink initial step size after the 3 run
    if (count >= 3) {
      tau = 0.01;
    }
    
    // Update count
    count += 1;
    
    // Update the sample side with ALS
    A1 = als_up(X_n, tnsr_list, 1, nng);
    
    // Update the feature and pixel-factor matrix using orthogonal features 
    A2 = wy_bls(X_n, A1, 2, tau, beta);
    A3 = wy_bls(X_n, A2, 3, tau, beta);
    
    tnsr_list = A3;
    if (count % 20 == 0) {
      cout << count << endl;  
    }
    
    // Track the change in the objective function
    all_res(count) = f_norm(X, vec_tensor(tnsr_list[0], tnsr_list[1], tnsr_list[2]));
    all_tol(count) = all_res(count-1) - all_res(count);
  }
  
  resmat = join_rows(all_tol, all_res);
  return List::create(_["A"] = tnsr_list[0], _["B"] = tnsr_list[1], 
                      _["C"] = tnsr_list[2], _["info"] = resmat);
}

// Algorithm for Uni-orthogonal Tensor Decomposition
// Orthogonal Should be applied on the feature side
// [[Rcpp::export]]
List uo_decomp (arma::cube X, List X_n, List tnsr_list, int rank, int iter, double tol, bool nng) {
  
  /* Initialization of parameters */
  mat resmat;                             // Store factorization result
  int count = 0;                          // Keep track of count
  vec all_res(iter+1), all_tol(iter+1); 
  List A1, A2, A3;
  double tau = 1, beta = 0.5;
  
  all_res(count) = all_tol(count) = f_norm(X, vec_tensor(tnsr_list[0], tnsr_list[1], tnsr_list[2]));
  
  /* Begin the update */
  while ((count != iter) && (all_res(count) > tol)) {
    
    // Shrink initial step size after the 3 run
    if (count >= 3) {
      tau = 0.01;
    }
    
    // Update count
    count += 1;
    
    // Update the factor matrices (orthogonal imposed on pixel side)
    A1 = als_up(X_n, tnsr_list, 1, nng);
    A2 = als_up(X_n, A1, 2, nng);
    A3 = wy_bls(X_n, A2, 3, tau, beta);
    
    tnsr_list = A3;
    if (count % 20 == 0) {
      cout << count << endl;  
    }
    
    // Track the change in the objective function
    all_res(count) = f_norm(X, vec_tensor(tnsr_list[0], tnsr_list[1], tnsr_list[2]));
    all_tol(count) = all_res(count-1) - all_res(count);
  }
  
  resmat = join_rows(all_tol, all_res);
  return List::create(_["A"] = tnsr_list[0], _["B"] = tnsr_list[1], 
                      _["C"] = tnsr_list[2], _["info"] = resmat);
}

// Algorithm for Nonnegative Tensor Decomposition
// [[Rcpp::export]]
List ntd (arma::cube X, List X_n, List tnsr_list, int rank, int iter, double tol) {
  
  /* Initialization of parameters */
  mat resmat;                             // Store factorization result
  int count = 0;                          // Keep track of count
  vec all_res(iter+1), all_tol(iter+1); 
  List A1, A2, A3;
  //double tau = 1, beta = 0.5;
  
  all_res(count) = all_tol(count) = f_norm(X, vec_tensor(tnsr_list[0], tnsr_list[1], tnsr_list[2]));
  
  /* Begin the update */
  while ((count != iter) && (all_res(count) > tol)) {
    
    // Update count
    count += 1;
    
    // Update all three factor matrices with NN-ALS
    A1 = als_up(X_n, tnsr_list, 1, true);
    A2 = als_up(X_n, A1, 2, true);
    A3 = als_up(X_n, A2, 3, true);
    
    tnsr_list = A3;
    if (count % 20 == 0) {
      cout << count << endl;  
    }
    
    // Track the change in the objective function
    all_res(count) = f_norm(X, vec_tensor(tnsr_list[0], tnsr_list[1], tnsr_list[2]));
    all_tol(count) = all_res(count-1) - all_res(count);
  }
  
  resmat = join_rows(all_tol, all_res);
  return List::create(_["A"] = tnsr_list[0], _["B"] = tnsr_list[1], 
                      _["C"] = tnsr_list[2], _["info"] = resmat);
}

// ALS algorithm for CP decomposition
// [[Rcpp::export]]
List cp_als (arma::cube tens, List modes, int rank, int iter, double thres) {
  
  // Obtain the mode-n unfolding of the tensor
  // List modes = unfold_ten(tens);
  mat X_1 = modes(0); mat X_2 = modes(1); mat X_3 = modes(2);
  
  // Extract dimension of the tensor
  int I = tens.n_rows;
  int J = tens.n_cols;
  int K = tens.n_slices;
  
  int count = 0; // Keep tracks of iteration
  vec f_err(iter+1); 
  vec tol(iter+1);
  
  if (rank > min({I, J, K})) {
    throw "Rank larger than at least one of the dimensions of the input tensor.";
  }
  
  // Initialize the factor matrices 
  mat A = randn<mat>(I,rank);
  mat B = randn<mat>(J,rank);
  mat C = randn<mat>(K,rank);
  f_err(count) = f_norm(tens, vec_tensor(A, B, C));
  tol(count) = 1;
  
  // Find the 3 factor matrix using ALS
  while ((count < iter)) {
    
    count += 1;
    
    A = X_1 * (krao_prod(C, B)) * pinv((C.t()*C) % (B.t()*B));
    B = X_2 * (krao_prod(C, A)) * pinv((C.t()*C) % (A.t()*A));
    C = X_3 * (krao_prod(B, A)) * pinv((B.t()*B) % (A.t()*A));
    
    f_err(count) = f_norm(tens, vec_tensor(A, B, C));  
    tol(count) = f_err(count-1) - f_err(count);
  }
  
  mat res = join_rows(f_err, tol);
  
  return List::create(_["A"] = A, _["B"] = B, _["C"] = C, _["info"] = res);
}

// Compute the gram matrix of a given layer
// [[Rcpp::export]]
arma::mat gramat (arma::mat layer) {
  
  int n = layer.n_rows;
  mat resmat(n, n, fill::zeros);
  
  for (int i = 0; i < n; i++) {
    for (int j = i; j < n; j++) {
      resmat(i,j) = as_scalar(layer.row(i) * layer.row(j).t());
    }
  }
  
  return resmat;
}

