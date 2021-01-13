#include <RcppArmadillo.h>
#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
arma::vec get_fiber(int R, arma::cube& B_out, arma::mat& Cim){
  arma::vec fiber(R); fiber.zeros();
  for (int r=0; r<R; r++){
    arma::mat Bjnr = B_out(arma::span::all, arma::span::all, arma::span(r));
    fiber(r) = arma::accu(Bjnr%Cim);
  }
  return fiber;
}

// [[Rcpp::export]]
Rcpp::List get_fiber_cum(int R, arma::cube& B_out, arma::mat& Cim, arma::vec ai, arma::vec am){
  arma::mat ABABim(size(Cim), arma::fill::zeros);
  arma::vec fiber(R); fiber.zeros();
  for (int r=0; r<R; r++){
    arma::mat Bjnr = B_out(arma::span::all, arma::span::all, arma::span(r));
    fiber(r) = arma::accu(Bjnr%Cim);
    ABABim += ai(r)*am(r)*Bjnr;
  }
  return Rcpp::List::create(Rcpp::Named("fiber")=fiber,
                            Rcpp::Named("ABABim")=ABABim);
}


// arma::cube mean_XX_L_n(arma::cube& X){
//   // estimate 1/L^2 sum(X_pq^im*X_pq^in)
//   int L1 = X.n_rows;
//   int L2 = X.n_cols;
//   int M = X.n_slices;
//   arma::mat hat_xi(M,M); hat_xi.zeros();
//   for (int j=0; j<M; j++){
//     for (int i=0; i<j; i++){
//       arma::mat tmp = X(arma::span::all,arma::span::all)
//     }
//   }
// }

// [[Rcpp::export]]
arma::mat ts_dot_v(arma::cube& X, arma::vec v){
  arma::mat Y(X.n_rows, X.n_cols);
  int L1 = X.n_cols;
  int L2 = X.n_rows;
  for (int i=0; i<L1; i++){
    for (int j=0; j<L2; j++){
      arma::rowvec tube_vec = X(arma::span(i), arma::span(j), arma::span::all);
      Y(i, j) = arma::as_scalar(tube_vec * v);
    }
  }
  return Y;
}
// [[Rcpp::export]]
arma::mat true_Sigma(arma::mat& Sigma_W, arma::mat& Sigma_Z, int i, int j){
  arma::mat resid_cov = Sigma_W % Sigma_W;
  resid_cov.diag() = Sigma_Z(i, i)*Sigma_Z(j, j) + Sigma_Z(i, j)*Sigma_Z(i, j) + (Sigma_Z(i,i)+Sigma_Z(j,j))*Sigma_W.diag() + Sigma_W.diag()%Sigma_W.diag();
  return resid_cov;
}

// [[Rcpp::export]]
arma::rowvec true_weight(arma::mat& Sigma_W, arma::mat& Sigma_Z, int i, int j){
  arma::mat resid_cov = Sigma_W % Sigma_W;
  resid_cov.diag() = Sigma_Z(i, i)*Sigma_Z(j, j) + Sigma_Z(i, j)*Sigma_Z(i, j) + (Sigma_Z(i,i)+Sigma_Z(j,j))*Sigma_W.diag() + Sigma_W.diag()%Sigma_W.diag();
  //resid_cov.print();
  //Sigma_W.print();
  arma::mat resid_cov_inv = arma::inv(resid_cov);
  //resid_cov_inv.print();
  arma::rowvec tmp = arma::sum(resid_cov_inv, 0);
  //tmp.print();
  double a = (1/arma::accu(tmp));
  //Rprintf("a=%f",a);
  arma::rowvec v = arma::as_scalar(a)*tmp;
  return v;
}
// [[Rcpp::export]]
arma::rowvec inv_weight(arma::mat& Sigma){
  arma::mat Sigma_inv = arma::inv(Sigma);
  arma::rowvec tmp = arma::sum(Sigma_inv, 0);
  double a = (1/arma::accu(tmp));
  arma::rowvec v = arma::as_scalar(a)*tmp;
  return v;
}
// [[Rcpp::export]]
arma::mat weighted_cov_true(arma::cube& X, arma::mat& Sigma_W, arma::mat& Sigma_Z){
  arma::mat Y(X.n_rows, X.n_cols);
  int L1 = X.n_cols;
  int L2 = X.n_rows;
  for (int i=0;i<L1;i++){
    for (int j=0;j<L2;j++){
      arma::rowvec tube_vec = X(arma::span(i), arma::span(j), arma::span::all);
      arma::rowvec v = true_weight(Sigma_W, Sigma_Z, i, j);
      Y(i, j) = arma::as_scalar(tube_vec * v.t());
    }
  }
  return Y;
}

// [[Rcpp::export]]
arma::mat weighted_cov_est(arma::cube& XX, arma::mat& mean_XX_L, arma::mat& est_Sigma_Z){
  arma::mat Y(XX.n_rows, XX.n_cols);
  int L1 = XX.n_cols;
  int L2 = XX.n_rows;
  arma::mat est_Sigma_XX_off = mean_XX_L%mean_XX_L;
  for (int i=0;i<L1;i++){
    for (int j=0;j<L2;j++){
      arma::rowvec tube_vec = XX(arma::span(i), arma::span(j), arma::span::all);
      arma::mat est_Sigma_XX = est_Sigma_XX_off;
      est_Sigma_XX.diag() += est_Sigma_Z(i,j)*est_Sigma_Z(i,j);
      arma::rowvec v = inv_weight(est_Sigma_XX);
      Y(i, j) = arma::as_scalar(tube_vec * v.t());
    }
  }
  return Y;
}

// [[Rcpp::export]]
arma::vec ind2subC(int ind, int L1, int L2){
  arma::vec sub(2);
  sub(0) = (ind - 1) % L1;
  sub(1) = int((ind - sub(0) - 1)) / L1;
  return sub;
}
// [[Rcpp::export]]
arma::rowvec get_tube(arma::cube X, int i, int j){
  arma::rowvec tube_vec = X(arma::span(i), arma::span(j), arma::span::all);
  return tube_vec;
}
// [[Rcpp::export]]
arma::mat CimCMat(arma::mat& Mat, int L1, int L2, int i, int m){
  arma::mat Cim(L2, L2);
  for (int j=0; j<L2; j++){
    for (int n=0; n<L2; n++){
      Cim(j, n) = Mat(j*L1+i, n*L2+m);
    }
  }
  return Cim;
}
// [[Rcpp::export]]
arma::mat CimCX(arma::cube X, int L2, int i, int m) {
  //arma::mat Cim(L2,L2);
  arma::mat Xi = X(arma::span(i), arma::span::all, arma::span::all);
  arma::mat Xm = X(arma::span(m), arma::span::all, arma::span::all);
  
  // for (int j=0; j<L2; j++){
  //   for (int n=0; n<L2; n++){
  //     double entry = arma::as_scalar(arma::cor( get_tube(X,i,j), get_tube(X,m,n)));
  //     Cim(j,n) = entry;
  //   }
  // }
  arma::mat Cim = arma::cor(Xi.t(),Xm.t());
  return Cim;
}
// [[Rcpp::export]]
arma::mat CimCX_cov(arma::cube X, int L2, int i, int m) {
  //arma::mat Cim(L2,L2);
  arma::mat Xi = X(arma::span(i), arma::span::all, arma::span::all);
  arma::mat Xm = X(arma::span(m), arma::span::all, arma::span::all);
  
  // for (int j=0; j<L2; j++){
  //   for (int n=0; n<L2; n++){
  //     double entry = arma::as_scalar(arma::cor( get_tube(X,i,j), get_tube(X,m,n)));
  //     Cim(j,n) = entry;
  //   }
  // }
  arma::mat Cim = arma::cov(Xi.t(),Xm.t());
  return Cim;
}
// [[Rcpp::export]]
arma::mat CimC(arma::mat X, arma::mat Y, int L2, int i, int m) {
  arma::mat Cim(L2,L2);
  int R = X.n_cols;
  for (int j=0; j<L2; j++){
    for (int n=0; n<L2; n++){
      double entry = 0;
      for (int r=0; r<R; r++){
        //Rcout << X(i,r) << std::endl;
        //Rf_PrintValue(X(i,r));
        entry += X(i,r)*X(m,r)*Y(j,r)*Y(n,r);
        //Rcout << entry << std::endl;
      }
      Cim(j,n) = entry;
    }
  }
  return Cim;
}

// [[Rcpp::export]]
Rcpp::List build_D(arma::mat X, arma::mat Y, arma::mat A, arma::mat B, arma::mat BtB) {
  int L1 = X.n_rows;
  int L2 = Y.n_rows;
  int R = X.n_cols;
  double diff = 0;
  // Rcpp::Timer timer;
  
  
  arma::cube D(L1, L1, R, arma::fill::zeros);
  
  arma::cube B_out(L2, L2, R);
  for (int r=0; r<R; r++){
    B_out.slice(r) = B.col(r) * B.col(r).t();
  }
  //arma::mat tmp = B.t() * B ;
  //arma::mat BtB = arma::inv(tmp%tmp);
  
  for (int m=0; m<L1; m++){
    for (int i=0; i<=m; i++){
      //Rcpp::Rcout << "m =" << std::endl;
      //Rcpp::Rcout << m << std::endl;
      //Rcpp::Rcout << "i =" << std::endl;
      //Rcpp::Rcout << i << std::endl;
      
      arma::mat Cim = CimC(X, Y, L2, i, m);
      arma::mat ABABim(L2, L2, arma::fill::zeros);
      arma::vec fiber(R);
      // timer.step("start");
      
      for (int r=0; r<R; r++){
        //Rcpp::Rcout << "r =" << std::endl;
        //Rcpp::Rcout << r << std::endl;
        arma::mat Bjnr = B_out.slice(r);
        
        fiber(r) = arma::as_scalar(arma::accu(Bjnr%Cim));
        ABABim += (A(i,r)*A(m,r))*Bjnr;
        // if (r==0){
        //   //Bjnr.print("Bjnr");
        //   Rcpp::Rcout << "A(i,r) =" << std::endl;
        //   Rcpp::Rcout << A(i,r) << std::endl;
        //   Rcpp::Rcout << "A(m,r) =" << std::endl;
        //   Rcpp::Rcout << A(m,r) << std::endl;
        //   ABABim.print("ABABim");
        // }
      }
      
      
      // timer.step("start");
      D.tube(i,m) = BtB*fiber;
      
      // timer.step("loop_R");
      diff += arma::accu(arma::square(ABABim-Cim));
      // Rcpp::NumericVector res(timer);
      // if ( (i==0) & (m==0)){
      //   Rcpp::Rcout << "loop_R =" << std::endl;
      //   Rcpp::Rcout << res[1]-res[0] << std::endl;
      //   goto STOP;
      // }
    }
  }
  // STOP:
  return Rcpp::List::create(Rcpp::Named("D")=D,
                            Rcpp::Named("diff")=diff);
}

// [[Rcpp::export]]
arma::mat GjnCX(arma::cube X, int L1, int j, int n) {
  arma::mat Xj = X(arma::span::all, arma::span(j), arma::span::all);
  arma::mat Xn = X(arma::span::all, arma::span(n), arma::span::all);
  arma::mat Gjn = arma::cor(Xj.t(),Xn.t());
  return Gjn;
}
// [[Rcpp::export]]
arma::mat GjnCMat(arma::mat& Mat, int L1, int L2, int j, int n){
  arma::mat Gjn(L1, L1);
  for (int i=0; i<L1; i++){
    for (int m=0; m<L1; m++){
      Gjn(i, m) = Mat(j*L1+i, n*L2+m);
    }
  }
  return Gjn;
}
// [[Rcpp::export]]
arma::mat GjnCX_cov(arma::cube X, int L1, int j, int n) {
  arma::mat Xj = X(arma::span::all, arma::span(j), arma::span::all);
  arma::mat Xn = X(arma::span::all, arma::span(n), arma::span::all);
  arma::mat Gjn = arma::cov(Xj.t(),Xn.t());
  return Gjn;
}
// [[Rcpp::export]]
arma::mat GjnC(arma::mat X, arma::mat Y, int L1, int j, int n) {
  arma::mat Gjn(L1,L1);
  int R = X.n_cols;
  for (int i=0; i<L1; i++){
    for (int m=0; m<L1; m++){
      double entry = 0;
      for (int r=0; r<R; r++){
        //Rcout << X(i,r) << std::endl;
        //Rf_PrintValue(X(i,r));
        entry += X(i,r)*X(m,r)*Y(j,r)*Y(n,r);
        //Rcout << entry << std::endl;
      }
      Gjn(i,m) = entry;
    }
  }
  return Gjn;
}
// [[Rcpp::export]]
arma::cube build_G(arma::mat X, arma::mat Y, arma::mat A_new, arma::mat B, arma::mat AtA) {
  int L1 = X.n_rows;
  int L2 = Y.n_rows;
  int R = X.n_cols;
  arma::cube G(L2, L2, R, arma::fill::zeros);
  arma::cube A_out(L1, L1, R);
  for (int r=0; r<R; r++){
    A_out.slice(r) = A_new.col(r) * A_new.col(r).t();
  }
  for (int n=0; n<L2; n++){
    for (int j=0; j<=n; j++){
      
      arma::mat Gjn = GjnC(X, Y, L1, j, n);
      
      arma::vec fiber(R);
      for (int r=0; r<R; r++){
        arma::mat Aimr = A_out.slice(r);
        fiber(r) = arma::as_scalar(arma::accu(Aimr%Gjn));
      }
      
      
      // timer.step("start");
      G.tube(j,n) = AtA*fiber;
      
      // timer.step("loop_R");
      // Rcpp::NumericVector res(timer);
      // if ( (i==0) & (m==0)){
      //   Rcpp::Rcout << "loop_R =" << std::endl;
      //   Rcpp::Rcout << res[1]-res[0] << std::endl;
      //   goto STOP;
      // }
    }
  }
  // STOP:
  return G;
}

// [[Rcpp::export]]
Rcpp::List symmetric_decompC_test(arma::mat X, arma::mat Y, int R, double tol=1e-4, int MAX_ITER=50){
  int L1 = X.n_rows;
  int L2 = Y.n_rows;
  arma::mat A = arma::randn(L1, R);
  arma::mat B = arma::randn(L2, R);
  int iter = 1;
  int iter0;
  double RMSE = 1;
  while ( (RMSE>tol) & (iter<=MAX_ITER) ){
    arma::mat tmp = B.t()*B;
    arma::mat BtB = arma::inv(tmp%tmp);
    Rcpp::List res = build_D(X, Y, A, B, BtB);
    arma::cube D = res["D"];
    double diff = res["diff"];
    RMSE = sqrt((diff)/(L1*L1*L2*L2));
    Rprintf("The RMSE of the %d-th step is: %2.3f \n ", iter-1,RMSE);
    //Rcpp::Rcout << "The RMSE of the" << iter-1 << "-th step is: " << RMSE <<"\n";
    
    arma::mat A_new(L1, R, arma::fill::zeros);
    for (int r=0; r<R; r++){
      arma::mat Er = symmatu(D.slice(r));
      arma::vec eigval;
      arma::mat eigvec;
      arma::eig_sym(eigval, eigvec, Er);
      Er.print();
      A_new.col(r) = sqrt(arma::as_scalar(eigval.tail(1)))*eigvec.tail_cols(1);
      eigval.print();
      eigval.tail(1).print();
      eigvec.print();
      eigvec.tail_cols(1).print();
    }
    if ((RMSE<tol) || (iter==(MAX_ITER-1))){
      iter0 = iter-1;
      break;
    }
    
    // Fix A, update B
    arma::mat tmp2 = A_new.t()*A_new;
    arma::mat AtA = arma::inv(tmp2%tmp2);
    arma::cube G = build_G(X, Y, A_new, B, AtA);
    arma::mat B_new(L2, R, arma::fill::zeros);
    for (int r=0; r<R; r++){
      arma::mat Fr = symmatu(G.slice(r));
      arma::vec eigval;
      arma::mat eigvec;
      arma::eig_sym(eigval, eigvec, Fr);
      B_new.col(r) = sqrt(arma::as_scalar(eigval.tail(1)))*eigvec.tail_cols(1);
    }
    A = A_new;
    B = B_new;
    iter0 = iter;
    iter += 1;
  }
  return Rcpp::List::create(Rcpp::Named("A")=A,
                            Rcpp::Named("B")=B,
                            Rcpp::Named("iter")=iter0,
                            Rcpp::Named("RMSE")=RMSE);
}

// [[Rcpp::export]]
Rcpp::List zero_non_zero_summary(arma::mat A0, arma::mat B0, arma::mat A, arma::mat B){
  int L1 = A0.n_rows;
  int L2 = B0.n_rows;
  double zero_MSE = 0;
  double non_zero_MSE = 0;
  double zero_average_F_norm = 0;
  double non_zero_average_F_norm = 0;
  int num_zero = 0;
  int num_non_zero = 0;
  for (int p=0; p<L1; p++){
    for (int q=0; q<L2; q++){
      for (int s=0; s<L1; s++){
        for (int t=0; t<L1; t++){
          if (!((p == s) & (q == t))){
            double C0 = arma::sum(A0.row(p)%B0.row(q)%A0.row(s)%B0.row(t));
            double C1 = arma::sum(A.row(p)%B.row(q)%A.row(s)%B.row(t));
            if (C0 == 0){
              num_zero += 1;
              zero_MSE += (C1)*(C1);
              zero_average_F_norm += (C1)*(C1);
            } else {
              num_non_zero += 1;
              non_zero_MSE += (C1-C0)*(C1-C0);
              non_zero_average_F_norm += (C1)*(C1);
            }
          }
        }
      }
    }
  }
  non_zero_average_F_norm /= num_non_zero;
  non_zero_MSE /= num_non_zero;
  zero_average_F_norm /= num_zero;
  zero_MSE /= num_zero;
  return Rcpp::List::create(Rcpp::Named("non_zero_average_F_norm")=non_zero_average_F_norm,
                            Rcpp::Named("non_zero_MSE")=non_zero_MSE,
                            Rcpp::Named("zero_average_F_norm")=zero_average_F_norm,
                            Rcpp::Named("zero_MSE")=zero_MSE);
}

// [[Rcpp::export]]
arma::cube sim_single_block(int L, int M, double rho, double alpha){
  int L1 = alpha * L;
  arma::cube W = arma::randn(L, L, M);
  arma::vec z = arma::randn(M);
  arma::cube X = W;
  for (int i=0; i<L1; i++){
    for (int j=0; j<L1; j++){
      arma::colvec tmp = sqrt(1-rho) * W.tube(i,j);
      X.tube(i, j) = sqrt(rho) * z + tmp;
    }
  }
  return X;
}

// [[Rcpp::export]]
Rcpp::List symmetric_decompXC(arma::cube X, int R, int L1, int L2, double tol=1e-4, int MAX_ITER=50, int type=1){
  //type=1: cor =2:cov
  arma::mat A = arma::randn(L1, R);
  arma::mat B = arma::randn(L2, R);
  int iter = 1;
  int iter0;
  double RMSE = 999;
  double RMSE2 = 999;
  double RMSE_old = 999;
  double RMSE_old2 = 999;
  while( (RMSE>tol) & (iter <= MAX_ITER)){
    arma::cube D(L1, L1, R);
    arma::mat tmp = B.t()*B;
    arma::mat BtB = arma::pinv(tmp%tmp);
    double diff = 0;
    double diff2 = 0;
    arma::cube B_out(L2, L2, R);
    for (int r=0; r<R; r++){
      B_out.slice(r) = B.col(r) * B.col(r).t();
    }
    for (int m=0; m<L1; m++){
      for (int i=0; i<=m; i++){
        arma::mat Cim;
        if (type == 2){
          Cim = CimCX_cov(X, L2, i, m);
        }
        else{
          Cim = CimCX(X, L2, i, m);
        }
        // TODO: the na case
        arma::mat ABABim(L2, L2, arma::fill::zeros);
        arma::vec fiber(R);
        for (int r=0; r<R; r++){
          //Rcpp::Rcout << "r =" << std::endl;
          //Rcpp::Rcout << r << std::endl;
          arma::mat Bjnr = B_out.slice(r);
          fiber(r) = arma::as_scalar(arma::accu(Bjnr % Cim));
          ABABim += (A(i,r)*A(m,r))*Bjnr;
          // if (r==0){
          //   //Bjnr.print("Bjnr");
          //   Rcpp::Rcout << "A(i,r) =" << std::endl;
          //   Rcpp::Rcout << A(i,r) << std::endl;
          //   Rcpp::Rcout << "A(m,r) =" << std::endl;
          //   Rcpp::Rcout << A(m,r) << std::endl;
          //   ABABim.print("ABABim");
          // }
        }
        // timer.step("start");
        D.tube(i,m) = BtB*fiber;
        
        // timer.step("loop_R");
        diff += arma::accu(arma::square( ABABim- Cim))/L1/L1/L2/L2;
        
        // Rcpp::NumericVector res(timer);
        // if ( (i==0) & (m==0)){
        //   Rcpp::Rcout << "loop_R =" << std::endl;
        //   Rcpp::Rcout << res[1]-res[0] << std::endl;
        //   goto STOP;
        // }
        if (i == m){
          arma::mat tmp= ABABim - Cim;
          tmp.diag().zeros();
          diff2 += arma::accu(arma::square(tmp))/(L1*L1*L2*L2-L1*L2);
        }
        else{
          diff2 += arma::accu(arma::square(ABABim-Cim))/(L1*L1*L2*L2-L1*L2);
        }
      }
    }
    
    
    RMSE = sqrt((diff));
    RMSE2 = sqrt(diff2);
    double RMSE_change = RMSE - RMSE_old;
    double RMSE_change2 = RMSE2 - RMSE_old2;
    if ( fabs(RMSE_change)< tol || fabs(RMSE_change2)< tol){
      // Rprintf("change stable!\n");
      iter0 = iter -1;
      break;
    }
    RMSE_old = RMSE;
    RMSE_old2 = RMSE2;
    arma::mat A_new(L1, R, arma::fill::zeros);
    for (int r=0; r<R; r++){
      arma::mat Er = symmatu(D.slice(r));
      // Er.print();
      arma::vec eigval;
      arma::mat eigvec;
      arma::eig_sym(eigval, eigvec, Er);
      // eigval.print();
      if (as_scalar(eigval.tail(1)) < 0.001){
        Er = -Er;
        // Rprintf("Er Negative \n");
        arma::eig_sym(eigval, eigvec, Er);
      }
      // Er.print();
      A_new.col(r) = sqrt(arma::as_scalar(eigval.tail(1)))*eigvec.tail_cols(1);
      // eigval.print();
      // eigval.tail(1).print();
      // eigvec.print();
      // eigvec.tail_cols(1).print();
    }
    if ((RMSE<tol) || (iter==(MAX_ITER-1))){
      iter0 = iter-1;
      // Rprintf("converge! \n");
      break;
    }
    
    
    // Fix A, update B
    arma::mat tmp2 = A_new.t()*A_new;
    arma::mat AtA = arma::pinv(tmp2%tmp2);
    arma::cube G(L2, L2, R, arma::fill::zeros);
    arma::cube A_out(L1, L1, R);
    for (int r=0; r<R; r++){
      A_out.slice(r) = A_new.col(r) * A_new.col(r).t();
    }
    for (int n=0; n<L2; n++){
      for (int j=0; j<= n; j++){
        arma::mat Gjn;
        if (type == 2){
          Gjn = GjnCX_cov(X, L1, j, n);
        }
        else{
          Gjn = GjnCX(X, L1, j, n);
        }
        arma::vec fiber(R);
        for (int r=0; r<R; r++){
          arma::mat Aimr = A_out.slice(r);
          fiber(r) = arma::as_scalar(arma::accu(Aimr%Gjn));
        }
        // timer.step("start");
        G.tube(j,n) = AtA*fiber;
        
        // timer.step("loop_R");
        // Rcpp::NumericVector res(timer);
        // if ( (i==0) & (m==0)){
        //   Rcpp::Rcout << "loop_R =" << std::endl;
        //   Rcpp::Rcout << res[1]-res[0] << std::endl;
        //   goto STOP;
        // }
        // TODO: na case
        
      }
    }
    arma::mat B_new(L2, R, arma::fill::zeros);
    for (int r=0; r<R; r++){
      arma::mat Fr = symmatu(G.slice(r));
      // Fr.print();
      arma::vec eigval;
      arma::mat eigvec;
      arma::eig_sym(eigval, eigvec, Fr);
      // eigval.print();
      if (as_scalar(eigval.tail(1)) < 0.001){
        Fr = -Fr;
        // Rprintf("Fr Negative \n");
        arma::eig_sym(eigval, eigvec, Fr);
      }
      B_new.col(r) = sqrt(arma::as_scalar(eigval.tail(1)))*eigvec.tail_cols(1);
    }
    A = A_new;
    B = B_new;
    iter0 = iter;
    iter += 1;
    // Rcpp::Rcout << iter << std::endl;
    // Rcpp::Rcout << RMSE << std::endl;
  }
  return Rcpp::List::create(Rcpp::Named("A")=A,
                            Rcpp::Named("B")=B,
                            Rcpp::Named("iter")=iter0,
                            Rcpp::Named("RMSE")=RMSE,
                            Rcpp::Named("RMSE2")=RMSE2);
}


