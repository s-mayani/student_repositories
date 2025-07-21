library("ggplot2")
library("scales")
library("akima")
library("pracma")
theme_set(theme_bw(base_size = 15))

data <- read.csv("fig_gen/introduction/field.csv", sep=",")



n <- 50
xout <- rep(linspace(1.001,2.99,n), each=n)
yout <- rep(linspace(1.001,2.99,n), n)

vals <- interpp(x=data$x, y=data$y, z=data$vx, xo=xout,yo=yout)
vals2 <- interpp(x=data$x, y=data$y, z=data$vy, xo=xout,yo=yout)

vals$z2 = vals2$z

df <- data.frame(x=vals$x,y=vals$y, vx=vals$z, vy=vals$z2)

f <- 0.05
ggplot(data=df, aes(x=x, y=y)) +
    geom_segment(aes(xend=x+f*vx, yend=y+f*vy), arrow = arrow(length = unit(0.1,"cm")))



f <- function(x,y) {
    vals <- interpp(x=data$x, y=data$y, z=data$vx, xo=x,yo=y)
    vals2 <- interpp(x=data$x, y=data$y, z=data$vy, xo=x,yo=y)
    c(vals$z, vals2$z)
}


vector_field <- function(
  f,  # Function describing the vector field
  xmin=0, xmax=1, ymin=0, ymax=1,
  width=600, height=600,
  iterations=50,
  epsilon=.01,
  trace=TRUE
) {
  z <- matrix(runif(width*height),nr=height)
  i_to_x <- function(i) xmin + i / width  * (xmax - xmin)
  j_to_y <- function(j) ymin + j / height * (ymax - ymin)
  x_to_i <- function(x) pmin( width,  pmax( 1, floor( (x-xmin)/(xmax-xmin) * width  ) ) )
  y_to_j <- function(y) pmin( height, pmax( 1, floor( (y-ymin)/(ymax-ymin) * height ) ) )
  i <- col(z)
  j <- row(z)
  x <- i_to_x(i)
  y <- j_to_y(j)
  res <- z
  for(k in 1:iterations) {
    v <- matrix( f(x, y), nc=2 )
    x <- x+.01*v[,1]
    y <- y+.01*v[,2]
    i <- x_to_i(x)
    j <- y_to_j(y)
    res <- res + z[cbind(i,j)]
    if(trace) {
      cat(k, "/", iterations, "\n", sep="")
      dev.hold()
      image(res)
      dev.flush()
    }
  }
  if(trace) {
    dev.hold()
    image(res>quantile(res,.6), col=0:1)
    dev.flush()
  }
  res
}

resolution <- 300

res <- vector_field(
  f,
  xmin=1.2, xmax=2.8, ymin=1.2, ymax=2.8,
  width=resolution, height=resolution,
  iterations=20,
  epsilon=.01
)

x <- linspace(1.2, 2.8, resolution)
y <- linspace(1.2, 2.8, resolution)

png(filename="figures/introduction/hero.png")
par(mar=c(0,0,0,0))
image(x, y, res, axes = FALSE, ann=FALSE,col=hcl.colors(300, "OrRd", rev=TRUE))
dev.off()


image(x, y, res, axes = FALSE, ann=FALSE,col=hcl.colors(300, "OrRd", rev=TRUE))


