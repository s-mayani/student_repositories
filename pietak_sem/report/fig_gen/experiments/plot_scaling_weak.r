library(ggplot2)
library(ggthemes)
library(Hmisc)
library(scales)
theme_set(theme_bw(base_size = 15))
o_color <- "#619cff"
o_shape <- 16

gg_color_hue <- function(n) {
  hues = seq(15, 375, length = n + 1)
  hcl(h = hues, l = 65, c = 100)[1:n]
}

colors <- gg_color_hue(3)

d <- data.frame(x=c(1:3)**2,
                y=rep(1,3))

d_2d <- read.csv("fig_gen/experiments/data_weak_2d_v2.csv", header=TRUE, sep="", comment.char="#")
d_2d$base = as.character(round(d_2d$num_nodes/(d_2d$num_ranks)^(1/2)))
d_2d <- d_2d[as.numeric(d_2d$base) > 2048,]
starts <- aggregate(time ~ num_nodes, data = d_2d[d_2d$num_ranks==1,], FUN = mean)
d_2d["efficiency"] = c(1:nrow(d_2d))
for (k in unique(d_2d$num_nodes)) {
    d_2d[round(d_2d$num_nodes/(d_2d$num_ranks)^(1/2)) == k,]$efficiency = d_2d[d_2d$num_ranks==1 & d_2d$num_nodes == k,]$time / (d_2d[round(d_2d$num_nodes/(d_2d$num_ranks)^(1/2)) == k,]$time)
}

latex(starts, file="figures/experiments/scaling_weak_2d.tex", colheads=c("Num nodes", "Time [s]"), booktabs=TRUE, rowname=NULL, table.env=FALSE, col.just=c("c","c")) 
p_2d <- ggplot() +
    stat_summary(d_2d, mapping=aes(num_ranks, efficiency, fill=base), fun.data=mean_cl_boot, geom="ribbon", alpha=0.3) +
    stat_summary(d_2d, mapping=aes(num_ranks, efficiency, color=base), fun=mean, geom="line", linewidth=0.75) +
    stat_summary(d_2d, mapping=aes(num_ranks, efficiency, color=base, shape=base), fun.y=mean, geom="point", size=1.75) +
    geom_line(d, mapping=aes(x, y), linetype="dashed") +
    stat_summary(d_2d, mapping=aes(num_ranks, efficiency, label=round(after_stat(y),2),color=base), fun=mean, geom="label", show.legend=FALSE, vjust = -0.3) +
    scale_x_continuous(trans="log2")+
    scale_color_manual(breaks = sort(as.numeric(unique(d_2d$base))), values=colors, labels = paste(as.character(sort(as.numeric(unique(d_2d$base)))), "²", sep="")) +
    scale_fill_manual(breaks = sort(as.numeric(unique(d_2d$base))), values=colors, labels = paste(as.character(sort(as.numeric(unique(d_2d$base)))), "²", sep="")) +
    scale_shape_discrete(breaks = sort(as.numeric(unique(d_2d$base))), labels = paste(as.character(sort(as.numeric(unique(d_2d$base)))), "²", sep="")) +
    scale_y_continuous(trans="log2") +
    labs(color = "Base",
         fill = "Base",
         shape = "Base") +
    xlab("Num ranks") +
    ylab("Efficiency")
p_2d

ggsave("figures/experiments/scaling_weak_2d.pdf", p_2d, width=15, heigh=15, units="cm")



d_3d <- read.csv("fig_gen/experiments/data_weak_3d_v2.csv", header=TRUE, sep="", comment.char="#")
d_3d$base = as.character(round(d_3d$num_nodes/(d_3d$num_ranks)^(1/3)))
d_3d <- d_3d[as.numeric(d_3d$base) > 64,]
starts <- aggregate(time ~ num_nodes, data = d_3d[d_3d$num_ranks==1,], FUN = mean)
d_3d["efficiency"] = c(1:nrow(d_3d))
for (k in unique(d_3d$num_nodes)) {
    d_3d[round(d_3d$num_nodes/(d_3d$num_ranks)^(1/3)) == k,]$efficiency = d_3d[d_3d$num_ranks==1 & d_3d$num_nodes == k,]$time / (d_3d[round(d_3d$num_nodes/(d_3d$num_ranks)^(1/3)) == k,]$time)
}



latex(starts, file="figures/experiments/scaling_weak_3d.tex", colheads=c("Num nodes", "Time [s]"), booktabs=TRUE, rowname=NULL, table.env=FALSE, col.just=c("c","c")) 
p_3d <- ggplot() +
    stat_summary(d_3d, mapping=aes(num_ranks, efficiency, fill=base), fun.data=mean_cl_boot, geom="ribbon", alpha=0.3) +
    stat_summary(d_3d, mapping=aes(num_ranks, efficiency, color=base), fun.y=mean, geom="line", linewidth=0.75) +
    stat_summary(d_3d, mapping=aes(num_ranks, efficiency, color=base, shape=base), fun.y=mean, geom="point", size=1.75) +
    geom_line(d, mapping=aes(x, y), linetype="dashed") +
    stat_summary(d_3d, mapping=aes(num_ranks, efficiency, label=round(after_stat(y),2),color=base), fun=mean, geom="label", show.legend=FALSE, vjust = -0.3) +
    scale_x_continuous(trans="log2")+
    scale_color_manual(breaks = sort(as.numeric(unique(d_3d$base))), values=colors, labels = paste(as.character(sort(as.numeric(unique(d_3d$base)))), "³", sep="")) +
    scale_fill_manual(breaks = sort(as.numeric(unique(d_3d$base))), values=colors, labels = paste(as.character(sort(as.numeric(unique(d_3d$base)))), "³", sep="")) +
    scale_shape_discrete(breaks = sort(as.numeric(unique(d_3d$base))), labels = paste(as.character(sort(as.numeric(unique(d_3d$base)))), "³", sep="")) +
    scale_y_continuous(trans="log2") +
    labs(color = "Base",
         fill = "Base",
         shape = "Base") +
    xlab("Num ranks") +
    ylab("Efficiency")
p_3d

ggsave("figures/experiments/scaling_weak_3d.pdf", p_3d, width=15, heigh=15, units="cm")




