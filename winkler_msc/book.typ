#set text(font: "Gulliver", size: 20pt)
#show heading : it => {
  if it.level == 1{
    text(size: 1.5em, it)
  } else{
    it
  }
}
= Flaschentales
== Chapter one
Once upon a time, there was a programmer called flaschenholz. And he spoke: 
```sh
pkill -9 cargo
```
and then he wondered why there was a cargo.lock file.
#[
  #let dx = "dx"
  #let starstar = "^"
  #let u0 = $u_0$
  #let u1 = $u_1$
  #let u2 = $u_2$
  #let u3 = $u_3$
  #let u4 = $u_4$
  #show "dx" : it => {"fugg"}
$ x**4*(-6* dx **3*u1 + 11*dx**2*u2 - 6*dx*u3 + u4)/(24*dx**4) + x**3*(8*dx**3*u1 - 14*dx**2*u2 + 7*dx*u3 - u4)/(6*dx**4) + x**2*(-12*dx**3*u1 + 19*dx**2*u2 - 8*dx*u3 + u4)/(4*dx**4) + x*(24*dx**3*u1 - 26*dx**2*u2 + 9*dx*u3 - u4)/(6*dx**4) + (24*dx**4*u0 - 50*dx**3*u1 + 35*dx**2*u2 - 10*dx*u3 + u4)/(24*dx**4) $
]