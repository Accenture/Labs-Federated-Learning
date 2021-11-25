import argparse
import plot.fig_shakespeare as shak
import plot.shakespeare as shak2
import plot.fig_quadratic_Decompo_Theo as q1
import plot.fig_quadratic_full_convergence as q2

parser = argparse.ArgumentParser(description="Experiment Parameters")

parser.add_argument(
    "--plot",
    type=str,
    help="type of figure",
    default="shak_paper",
)

parser.add_argument(
    "--show",
    type=lambda x: x == "True",
    help="show the figure",
    default=False,
)

args = parser.parse_args()


if args.plot == "quadratic":
    q1.quadra_Theo1(
        n=10, m=5, n_params=20, bound=1, eta_l=0.1, eta_g=1.0, K=10, n_draw=1000
    )
    q2.quadra_full_train(
        n_iter=1000, n=100, m=5, n_params=20, bound=1, eta_l=0.2, eta_g=1.0, K=1
    )


if args.plot == "shak_paper":
    """plot in manuscript with Shakespeare"""
    shak.shakespeare_paper(
        plot_name="shak_paper",
        metric="loss",
        samplings=["MD", "Uniform", "Clustered"],
        T=600,
        n_SGD=50,
        lr_l=1.5,
        B=64,
        n_seeds=30,
        show=args.show,
    )

    shak.shakespeare_paper_appendix(
        plot_name="shak_paper_appendix",
        metric="loss",
        samplings=["MD", "Uniform", "Clustered"],
        T=600,
        n_SGD=50,
        lr_l=1.5,
        B=64,
        n_seeds=30,
        show=args.show,
    )


elif args.plot == "shak_appendix_varying_m":
    """additional plots in appendix with 5 and 10% of clients for n=80"""
    shak2.shakespeare_paper_varying_m(
        plot_name=args.plot,
        metric="loss",
        samplings=["MD", "Uniform", "Clustered"],
        T=1000,
        n_SGD=50,
        lr_l=1.5,
        B=64,
        n_seeds=15,
        show=args.show,
    )
    shak2.VaryingM(
        plot_name=args.plot + "_inde",
        metric="loss",
        l_m=[4, 8],
        samplings=["MD", "Uniform", "Clustered"],
        T=1000,
        n_SGD=50,
        lr_l=1.5,
        B=64,
        n_seeds=15,
        show=args.show,
    )

elif args.plot == "shak_appendix_varying_K":

    shak2.shakespeare_paper_varying_K(
        plot_name=args.plot,
        metric="loss",
        samplings=["MD", "Uniform"],
        T=2500,
        n_SGD=1,
        lr_l=1.5,
        B=64,
        n_seeds=15,
        show=args.show,
    )

    shak2.VaryingK(
        plot_name=args.plot + "_inde",
        metric="loss",
        l_m=[8, 40],
        samplings=["MD", "Uniform"],
        T=2500,
        n_SGD=1,
        lr_l=1.5,
        B=64,
        n_seeds=15,
        show=args.show,
    )
