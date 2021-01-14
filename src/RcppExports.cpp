// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

// kro_prod
arma::mat kro_prod(arma::mat A, arma::mat B);
RcppExport SEXP _tnsrcomp_kro_prod(SEXP ASEXP, SEXP BSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type A(ASEXP);
    Rcpp::traits::input_parameter< arma::mat >::type B(BSEXP);
    rcpp_result_gen = Rcpp::wrap(kro_prod(A, B));
    return rcpp_result_gen;
END_RCPP
}
// krao_prod
arma::mat krao_prod(arma::mat A, arma::mat B);
RcppExport SEXP _tnsrcomp_krao_prod(SEXP ASEXP, SEXP BSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type A(ASEXP);
    Rcpp::traits::input_parameter< arma::mat >::type B(BSEXP);
    rcpp_result_gen = Rcpp::wrap(krao_prod(A, B));
    return rcpp_result_gen;
END_RCPP
}
// unfold_ten
List unfold_ten(arma::cube tens);
RcppExport SEXP _tnsrcomp_unfold_ten(SEXP tensSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::cube >::type tens(tensSEXP);
    rcpp_result_gen = Rcpp::wrap(unfold_ten(tens));
    return rcpp_result_gen;
END_RCPP
}
// fold_ten
arma::cube fold_ten(arma::mat A, arma::vec dim, const int mode);
RcppExport SEXP _tnsrcomp_fold_ten(SEXP ASEXP, SEXP dimSEXP, SEXP modeSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type A(ASEXP);
    Rcpp::traits::input_parameter< arma::vec >::type dim(dimSEXP);
    Rcpp::traits::input_parameter< const int >::type mode(modeSEXP);
    rcpp_result_gen = Rcpp::wrap(fold_ten(A, dim, mode));
    return rcpp_result_gen;
END_RCPP
}
// f_norm
float f_norm(arma::cube tens, arma::cube approx);
RcppExport SEXP _tnsrcomp_f_norm(SEXP tensSEXP, SEXP approxSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::cube >::type tens(tensSEXP);
    Rcpp::traits::input_parameter< arma::cube >::type approx(approxSEXP);
    rcpp_result_gen = Rcpp::wrap(f_norm(tens, approx));
    return rcpp_result_gen;
END_RCPP
}
// tens_mat
arma::cube tens_mat(arma::cube tens, arma::mat X, int mode);
RcppExport SEXP _tnsrcomp_tens_mat(SEXP tensSEXP, SEXP XSEXP, SEXP modeSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::cube >::type tens(tensSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type X(XSEXP);
    Rcpp::traits::input_parameter< int >::type mode(modeSEXP);
    rcpp_result_gen = Rcpp::wrap(tens_mat(tens, X, mode));
    return rcpp_result_gen;
END_RCPP
}
// core_ten
arma::cube core_ten(List core_mat);
RcppExport SEXP _tnsrcomp_core_ten(SEXP core_matSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< List >::type core_mat(core_matSEXP);
    rcpp_result_gen = Rcpp::wrap(core_ten(core_mat));
    return rcpp_result_gen;
END_RCPP
}
// find_core
arma::cube find_core(arma::cube tens, List mats);
RcppExport SEXP _tnsrcomp_find_core(SEXP tensSEXP, SEXP matsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::cube >::type tens(tensSEXP);
    Rcpp::traits::input_parameter< List >::type mats(matsSEXP);
    rcpp_result_gen = Rcpp::wrap(find_core(tens, mats));
    return rcpp_result_gen;
END_RCPP
}
// vec_tensor
arma::cube vec_tensor(arma::mat A, arma::mat B, arma::mat C);
RcppExport SEXP _tnsrcomp_vec_tensor(SEXP ASEXP, SEXP BSEXP, SEXP CSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type A(ASEXP);
    Rcpp::traits::input_parameter< arma::mat >::type B(BSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type C(CSEXP);
    rcpp_result_gen = Rcpp::wrap(vec_tensor(A, B, C));
    return rcpp_result_gen;
END_RCPP
}
// als_up
List als_up(List X_list, List tens_list, int mode, bool nng);
RcppExport SEXP _tnsrcomp_als_up(SEXP X_listSEXP, SEXP tens_listSEXP, SEXP modeSEXP, SEXP nngSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< List >::type X_list(X_listSEXP);
    Rcpp::traits::input_parameter< List >::type tens_list(tens_listSEXP);
    Rcpp::traits::input_parameter< int >::type mode(modeSEXP);
    Rcpp::traits::input_parameter< bool >::type nng(nngSEXP);
    rcpp_result_gen = Rcpp::wrap(als_up(X_list, tens_list, mode, nng));
    return rcpp_result_gen;
END_RCPP
}
// wy_bls
List wy_bls(List X_modes, List tens_list, int mode, float tau, float beta);
RcppExport SEXP _tnsrcomp_wy_bls(SEXP X_modesSEXP, SEXP tens_listSEXP, SEXP modeSEXP, SEXP tauSEXP, SEXP betaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< List >::type X_modes(X_modesSEXP);
    Rcpp::traits::input_parameter< List >::type tens_list(tens_listSEXP);
    Rcpp::traits::input_parameter< int >::type mode(modeSEXP);
    Rcpp::traits::input_parameter< float >::type tau(tauSEXP);
    Rcpp::traits::input_parameter< float >::type beta(betaSEXP);
    rcpp_result_gen = Rcpp::wrap(wy_bls(X_modes, tens_list, mode, tau, beta));
    return rcpp_result_gen;
END_RCPP
}
// bontf
List bontf(arma::cube X, List X_n, List tnsr_list, int rank, int iter, double tol, bool nng);
RcppExport SEXP _tnsrcomp_bontf(SEXP XSEXP, SEXP X_nSEXP, SEXP tnsr_listSEXP, SEXP rankSEXP, SEXP iterSEXP, SEXP tolSEXP, SEXP nngSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::cube >::type X(XSEXP);
    Rcpp::traits::input_parameter< List >::type X_n(X_nSEXP);
    Rcpp::traits::input_parameter< List >::type tnsr_list(tnsr_listSEXP);
    Rcpp::traits::input_parameter< int >::type rank(rankSEXP);
    Rcpp::traits::input_parameter< int >::type iter(iterSEXP);
    Rcpp::traits::input_parameter< double >::type tol(tolSEXP);
    Rcpp::traits::input_parameter< bool >::type nng(nngSEXP);
    rcpp_result_gen = Rcpp::wrap(bontf(X, X_n, tnsr_list, rank, iter, tol, nng));
    return rcpp_result_gen;
END_RCPP
}
// uo_decomp
List uo_decomp(arma::cube X, List X_n, List tnsr_list, int rank, int iter, double tol, bool nng);
RcppExport SEXP _tnsrcomp_uo_decomp(SEXP XSEXP, SEXP X_nSEXP, SEXP tnsr_listSEXP, SEXP rankSEXP, SEXP iterSEXP, SEXP tolSEXP, SEXP nngSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::cube >::type X(XSEXP);
    Rcpp::traits::input_parameter< List >::type X_n(X_nSEXP);
    Rcpp::traits::input_parameter< List >::type tnsr_list(tnsr_listSEXP);
    Rcpp::traits::input_parameter< int >::type rank(rankSEXP);
    Rcpp::traits::input_parameter< int >::type iter(iterSEXP);
    Rcpp::traits::input_parameter< double >::type tol(tolSEXP);
    Rcpp::traits::input_parameter< bool >::type nng(nngSEXP);
    rcpp_result_gen = Rcpp::wrap(uo_decomp(X, X_n, tnsr_list, rank, iter, tol, nng));
    return rcpp_result_gen;
END_RCPP
}
// ntd
List ntd(arma::cube X, List X_n, List tnsr_list, int rank, int iter, double tol);
RcppExport SEXP _tnsrcomp_ntd(SEXP XSEXP, SEXP X_nSEXP, SEXP tnsr_listSEXP, SEXP rankSEXP, SEXP iterSEXP, SEXP tolSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::cube >::type X(XSEXP);
    Rcpp::traits::input_parameter< List >::type X_n(X_nSEXP);
    Rcpp::traits::input_parameter< List >::type tnsr_list(tnsr_listSEXP);
    Rcpp::traits::input_parameter< int >::type rank(rankSEXP);
    Rcpp::traits::input_parameter< int >::type iter(iterSEXP);
    Rcpp::traits::input_parameter< double >::type tol(tolSEXP);
    rcpp_result_gen = Rcpp::wrap(ntd(X, X_n, tnsr_list, rank, iter, tol));
    return rcpp_result_gen;
END_RCPP
}
// cp_als
List cp_als(arma::cube tens, List modes, int rank, int iter, double thres);
RcppExport SEXP _tnsrcomp_cp_als(SEXP tensSEXP, SEXP modesSEXP, SEXP rankSEXP, SEXP iterSEXP, SEXP thresSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::cube >::type tens(tensSEXP);
    Rcpp::traits::input_parameter< List >::type modes(modesSEXP);
    Rcpp::traits::input_parameter< int >::type rank(rankSEXP);
    Rcpp::traits::input_parameter< int >::type iter(iterSEXP);
    Rcpp::traits::input_parameter< double >::type thres(thresSEXP);
    rcpp_result_gen = Rcpp::wrap(cp_als(tens, modes, rank, iter, thres));
    return rcpp_result_gen;
END_RCPP
}
// gramat
arma::mat gramat(arma::mat layer);
RcppExport SEXP _tnsrcomp_gramat(SEXP layerSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type layer(layerSEXP);
    rcpp_result_gen = Rcpp::wrap(gramat(layer));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_tnsrcomp_kro_prod", (DL_FUNC) &_tnsrcomp_kro_prod, 2},
    {"_tnsrcomp_krao_prod", (DL_FUNC) &_tnsrcomp_krao_prod, 2},
    {"_tnsrcomp_unfold_ten", (DL_FUNC) &_tnsrcomp_unfold_ten, 1},
    {"_tnsrcomp_fold_ten", (DL_FUNC) &_tnsrcomp_fold_ten, 3},
    {"_tnsrcomp_f_norm", (DL_FUNC) &_tnsrcomp_f_norm, 2},
    {"_tnsrcomp_tens_mat", (DL_FUNC) &_tnsrcomp_tens_mat, 3},
    {"_tnsrcomp_core_ten", (DL_FUNC) &_tnsrcomp_core_ten, 1},
    {"_tnsrcomp_find_core", (DL_FUNC) &_tnsrcomp_find_core, 2},
    {"_tnsrcomp_vec_tensor", (DL_FUNC) &_tnsrcomp_vec_tensor, 3},
    {"_tnsrcomp_als_up", (DL_FUNC) &_tnsrcomp_als_up, 4},
    {"_tnsrcomp_wy_bls", (DL_FUNC) &_tnsrcomp_wy_bls, 5},
    {"_tnsrcomp_bontf", (DL_FUNC) &_tnsrcomp_bontf, 7},
    {"_tnsrcomp_uo_decomp", (DL_FUNC) &_tnsrcomp_uo_decomp, 7},
    {"_tnsrcomp_ntd", (DL_FUNC) &_tnsrcomp_ntd, 6},
    {"_tnsrcomp_cp_als", (DL_FUNC) &_tnsrcomp_cp_als, 5},
    {"_tnsrcomp_gramat", (DL_FUNC) &_tnsrcomp_gramat, 1},
    {NULL, NULL, 0}
};

RcppExport void R_init_tnsrcomp(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
