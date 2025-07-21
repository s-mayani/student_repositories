library("ggplot2")
library("scales")
theme_set(theme_bw(base_size = 15))


data_trig_2d <- read.csv("fig_gen/experiments/performance_final_trig_2d.csv", sep="")
data_trig_3d <- read.csv("fig_gen/experiments/performance_final_trig_3d.csv", sep="")
data_poly_2d <- read.csv("fig_gen/experiments/performance_final_poly_2d.csv", sep="")
data_poly_3d <- read.csv("fig_gen/experiments/performance_final_poly_3d.csv", sep="")

ideal <- data.frame(x = 2^(4:10), y = 18./(2^(4:10))^2, w = 1./(2^(4:10)))


p_trig <- ggplot() +
    geom_point(data=data_trig_2d, aes(x=num_nodes, y=interp_error_coef, color="2D", shape="2D"),size=2.5) +
    geom_point(data=data_trig_3d, aes(x=num_nodes, y=interp_error_coef, color="3D", shape="3D"),size=2.5) +
    geom_line(data=ideal, aes(x=x, y=y,linetype="Ideal 1/n²"), ) +
    scale_x_continuous(trans = "log2") +
    scale_y_continuous(trans = "log2", breaks = trans_breaks("log2", function(x) 2^x), labels = trans_format("log2", math_format(2^.x)))+
    labs(x = "Number of nodes per axis",
         y = "Error",
         color = "Dimension",
         linetype = "Order",
         shape="Dimension") 
p_trig

ggsave("figures/experiments/convergence_trig.pdf", p_trig, width=15, heigh=15, units="cm")



p_poly <- ggplot() +
    geom_point(data=data_poly_2d, aes(x=num_nodes, y=interp_error_coef, color="2D", shape="2D"),size=2.5) +
    geom_point(data=data_poly_3d, aes(x=num_nodes, y=interp_error_coef, color="3D", shape="3D"),size=2.5) +
    geom_line(data=ideal, aes(x=x, y=y,linetype="Ideal 1/n²"), ) +
    scale_x_continuous(trans = "log2") +
    scale_y_continuous(trans = "log2", breaks = trans_breaks("log2", function(x) 2^x), labels = trans_format("log2", math_format(2^.x)))+
    labs(x = "Number of nodes per axis",
         y = "Error",
         color = "Dimension",
         linetype = "Order",
         shape="Dimension")
p_poly

ggsave("figures/experiments/convergence_poly.pdf", p_poly, width=15, heigh=15, units="cm")

