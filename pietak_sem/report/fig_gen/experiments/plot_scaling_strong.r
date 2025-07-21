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


loadData <- function(d_ss) {
    #d_ss <- read.csv(paste("fig_gen/experiments/",filename,sep=""), header=TRUE, sep="", comment.char="#")
    starts <- aggregate(time ~ num_nodes, data = d_ss[d_ss$num_ranks==1,], FUN = mean)
    d_ss["speedup"] = c(1:nrow(d_ss))
    print(unique(d_ss$num_nodes))
    print(starts)
    for (k in unique(d_ss$num_nodes)) {
        d_ss[d_ss$num_nodes == k,]$speedup = starts[starts$num_nodes==k,]$time / d_ss[d_ss$num_nodes == k,]$time
    }
    d_ss$num_nodes = as.character(d_ss$num_nodes)

    return(list(d_ss,starts))
}
d <- data.frame(x = c(1:8),
                y = c(1:8))

d_2d <- read.csv("fig_gen/experiments/data_strong_2d_v2.csv", header=TRUE, sep="", comment.char="#")
d_2d <- d_2d[d_2d$num_nodes < 32768 & d_2d$num_nodes >= 4096,]
t <- loadData(d_2d)
d_2d <- t[[1]]
s_2d <- t[[2]]
latex(s_2d, file="figures/experiments/scaling_strong_2d.tex", colheads=c("Num nodes", "Time [s]"), booktabs=TRUE, rowname=NULL, table.env=FALSE, col.just=c("c","c")) 
d_2d <- d_2d[order(d_2d$num_nodes),]
p_2d <- ggplot() +
    stat_summary(d_2d, mapping=aes(num_ranks, speedup, fill=num_nodes), fun.data=mean_cl_boot, geom="ribbon", alpha=0.3) +
    stat_summary(d_2d, mapping=aes(num_ranks, speedup, color=num_nodes), fun.y=mean, geom="line", linewidth=0.75) +
    stat_summary(d_2d, mapping=aes(num_ranks, speedup, color=num_nodes,, shape=num_nodes), fun.y=mean, geom="point", size=1.75) +
    geom_line(d, mapping=aes(x,y), linetype="dashed") +
    stat_summary(d_2d, mapping=aes(num_ranks, speedup, label=round(after_stat(y),2),color=num_nodes), fun=mean, geom="label", show.legend=FALSE, vjust = -0.3) +
    scale_x_continuous(trans="log2", breaks=c(1,2,4,8,16), labels = as.character(c(1,2,4,8,16)))+
    scale_color_manual(breaks = sort(as.numeric(unique(d_2d$num_nodes))),values = colors, labels = paste(as.character(sort(as.numeric(unique(d_2d$num_nodes)))), "²", sep="")) +
    scale_fill_manual(breaks = sort(as.numeric(unique(d_2d$num_nodes))),values = colors, labels = paste(as.character(sort(as.numeric(unique(d_2d$num_nodes)))), "²", sep="")) +
    scale_shape_discrete(breaks = sort(as.numeric(unique(d_2d$num_nodes))), labels = paste(as.character(sort(as.numeric(unique(d_2d$num_nodes)))), "²", sep="")) +
    scale_y_continuous(trans="log2") +
    labs(color = "Num nodes",
         fill = "Num nodes",
         shape = "Num nodes") +
    xlab("Num ranks") +
    ylab("Speedup")
p_2d

ggsave("figures/experiments/scaling_strong_2d.pdf", p_2d, width=15, heigh=15, units="cm")


d_3d <- read.csv("fig_gen/experiments/data_strong_3d_v2.csv", header=TRUE, sep="", comment.char="#")
d_3d <- d_3d[d_3d$num_nodes < 1024,]
t <- loadData(d_3d)
d_3d <- t[[1]]
s_3d <- t[[2]]
latex(s_3d, file="figures/experiments/scaling_strong_3d.tex", colheads=c("Num nodes", "Time [s]"), booktabs=TRUE, rowname=NULL, table.env=FALSE, col.just=c("c","c")) 
d_3d <- d_3d[order(d_3d$num_nodes),]
p_3d <- ggplot() +
    stat_summary(d_3d, mapping=aes(num_ranks, speedup, fill=num_nodes), fun.data=mean_cl_boot, geom="ribbon", alpha=0.3) +
    stat_summary(d_3d, mapping=aes(num_ranks, speedup, color=num_nodes), fun.y=mean, geom="line", linewidth=0.75) +
    stat_summary(d_3d, mapping=aes(num_ranks, speedup, color=num_nodes,, shape=num_nodes), fun.y=mean, geom="point", size=1.75) +
    geom_line(d, mapping=aes(x,y), linetype="dashed") +
    stat_summary(d_3d, mapping=aes(num_ranks, speedup, label=round(after_stat(y),2),color=num_nodes), fun=mean, geom="label", show.legend=FALSE, vjust = -0.3) +
    scale_x_continuous(trans="log2", breaks=c(1,2,4,8,16), labels = as.character(c(1,2,4,8,16)))+
    scale_color_manual(breaks = sort(as.numeric(unique(d_3d$num_nodes))),values = colors, labels = paste(as.character(sort(as.numeric(unique(d_3d$num_nodes)))), "³", sep="")) +
    scale_fill_manual(breaks = sort(as.numeric(unique(d_3d$num_nodes))), values = colors, labels = paste(as.character(sort(as.numeric(unique(d_3d$num_nodes)))), "³", sep="")) +
    scale_shape_discrete(breaks = sort(as.numeric(unique(d_3d$num_nodes))), labels = paste(as.character(sort(as.numeric(unique(d_3d$num_nodes)))), "³", sep="")) +
    scale_y_continuous(trans="log2") +
    labs(color = "Num nodes",
         fill = "Num nodes",
         shape = "Num nodes") +
    xlab("Num ranks") +
    ylab("Speedup")
p_3d

ggsave("figures/experiments/scaling_strong_3d.pdf", p_3d, width=15, heigh=15, units="cm")