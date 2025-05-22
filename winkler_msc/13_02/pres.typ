#import "@preview/polylux:0.3.1": *
#import "@preview/physica:0.9.2": *
#import "@preview/pinit:0.1.3": *
#import "@preview/cetz:0.2.0": canvas, plot
//#import "tests.typ"
//#import themes.simple: *
//#set math.equation.numbering("(1)")
#let authors = {"Author"}
#import themes.university: *


#show: university-theme.with(aspect-ratio: "4-3"
  //it => [it]
)

#let author_state = state("author", "John Doe")
#let coauthors_state = state("coauthors", "")
#let title_state = state("title", "Science")
#let bottomgraybar = place(dx: 0%, dy: 100% - 14pt)[#rect(height: 14pt, width: 100%, stroke: none, fill: rgb("ccccff"))[
  #set text(size: 10pt)
  #grid(rows: (1fr), columns: (1fr, 1fr, 1fr),
  align(horizon + left,pad(x: 5pt, y:0pt)[#text(text(weight: 900)[#author_state.display()] + text(weight: 300)[
        #coauthors_state.display()])]),
  align(horizon + center, pad(x: 0pt, y:0pt)[#title_state.display()]),
  align(horizon + right, pad(x: 5pt, y:0pt)[#datetime.today().display("[month repr:long] [day], [year]") #h(1cm) #text(logic.logical-slide.display() + [~/~] + utils.last-slide-number)]))
]]
#let equationbox(stroke: black, content) = box(width: 100%, align(center, box(stroke: 2pt + stroke, outset: 5pt, inset:5pt, radius: 5pt)[
  #content
]))
#let topgraybar(content) = place(dx: 0%, dy: 0%)[#rect(height: 60pt, width: 100%, stroke: none, fill: rgb("ccccff"))[
  #set text(size: 40pt)
  #align(horizon + left, h(5pt) + content)
]]
//#page(background: block(width: 100%, height: 100%)
#let titelsleid(title_image_height : 48%,showimage: false, author: "John Doe", email : "", coauthors : (), title : "Science", groupname : "SomeGroup") = {polylux-slide[
  #place(dx: 0%, dy: 6%, rect(stroke: 0pt, fill: rgb("dddddd"), width: 28%, height: title_image_height)[
    #place(dy: 7%)[#block(width: 100%, height: 100%, align(top + center, image("logos/PSI.png", width: 60%)))]
    #place(dy: 40%)[#block(width: 100%, height: 100%, align(top + center, image("logos/eth.png", width: 60%)))]
  ])
  #place(dx: 90%, dy: 6%, rect(stroke: 0pt, fill: rgb("dddddd"), width: 10%, height: title_image_height))
  #if showimage [
    #place(dx: 30%, dy: 6%, box(height: title_image_height)[
      #let tv = text(font: "Noto Serif", size: 15pt, "WIR SCHAFFEN WISSEN - HEUTE FÜR MORGEN")
      #image("logos/PSI_helicopter.jpg", height: 100%)
      #align(bottom + right, box(inset: 2pt, radius: 5pt, fill: rgb("cccccccc"), tv))
    ])
  ] else [
    #place(dx: 30%, dy: 6%, box(height: title_image_height)[
      #let tv = text(font: "Noto Serif", size: 15pt, "WIR SCHAFFEN WISSEN - HEUTE FÜR MORGEN")
      #rect(height: 100%, width: 15cm)
      #align(bottom + right, box(inset: 2pt, radius: 5pt, fill: rgb("cccccccc"), tv))
    ])
  ]
  #if email.len() > 0 [
    #place(dy: 88%)[
      #box(width: 100%)[
        #align(right)[
          #text(size: 13pt)[Email: #text(font: "Victor Mono", weight: 900)[#email]] #h(15%)
        ]
      ]
    ]
  ]

  #let thing(body) = style(styles => {
    let size = measure(body, styles)
    text(size: 6pt)[Width of "#body" is #size.width]
  })
  
  #author_state.update(author)
  #title_state.update(title)
  #{
    let n = 0
    let str = ""
    while n < coauthors.len() {
      str = str + ", " + coauthors.at(n)
      n = n + 1
    }
    coauthors_state.update(str)
    text(size: 15pt, weight: 300)[]
  }
  #bottomgraybar
  //#place(dx: 0%, dy: 0%, rect(stroke: 0pt, fill: gray, width: 200pt, height: 50pt))
  
//])[
  #grid(rows:(4fr, 2fr), columns: (1fr))[][
    #place(dx: 15%)[
      #stack(spacing: 30pt,
        block(text(size: 15pt, weight: 900)[#author_state.display()] + text(size: 15pt, weight: 300)[#coauthors_state.display()] + text(size: 15pt, weight: 300)[#h(10pt) :: #h(10pt) #groupname]),
        text()[#title_state.display()],
        text(size: 15pt)[#datetime.today().display("[month repr:long] [day], [year]")]
      )
    ]
  ]
]}
#let normalislide(padding: 3cm, title : "", content) = {
  polylux-slide()[
    #bottomgraybar
    #topgraybar()[#title]
    #pad(x: padding, block(width: 100%, height: 100%, align(horizon, content)))
  ]
}
#let newsectionslide(title : "", content) = {
  utils.register-section(title)
  polylux-slide()[
    #bottomgraybar
    #topgraybar()[#title]
    #pad(x: 4cm, block(width: 100%, height: 100%)[
      #show title: it => {text(weight: 900, fill: rgb("5555bb"), it)}
      #align(horizon, polylux-outline())
    ])
  ]
  
}
#let newsectionslide_nocontents(title : "", content) = {
  utils.register-section(title)
  polylux-slide()[
    #topgraybar()[#title]
    #pad(x: 4cm, block(width: 100%, height: 100%)[
      #align(horizon, content)
    ])
  ]
}
#let sections-state = state("polylux-sections", ())
#let cool-outline(enum-args: (:), spacing : 1cm, padding: 0pt) = locate( loc => {
  let sections = sections-state.final(loc)
  pad(padding, enum(
    ..enum-args,
    ..sections.map(section => link(section.loc, section.body + v(spacing)))
  ))
})
//#titelsleid(showimage: true, title_image_height: 50%, author: "Pranas Astrauskas", coauthors : ("S. Heinekamp", "A. Adelmann"), title : "Variable selection for Bayesian calibration", groupname : "AMAS Group, LSM")
#titelsleid(title_image_height: 50%, author: "Manuel Winkler", email: "flaschenholz@protonmail.com", coauthors : ("S. Mayani", "A. Adelmann"), title : "Undulators et al.", groupname : "AMAS Group, LSM")

//#normalislide(title: "Outline")[
//  #cool-outline(spacing : 10pt)
//]
#normalislide(title: "Lab Undulator fields")[
  #set text(17pt)
  #set math.vec(delim: "[",)
  #block(width: 100%)[
    The electric and magnetic field of an Undulator module are defined by
    //#align(center, block(stroke: 3pt, radius: 10pt, inset: 10pt)[
      $ vb(B)_U &= cases(vec(0, B_0 cosh(k_u y) k_u z e^(-(k_u z)^2/2), B_0 sinh(k_u y) e^(-(k_u z)^2/2)) "if" z < 0, vec(0, B_0 cosh(k_u y) sin(k_u z),B_0 sinh(k_u y) cos(k_u z)) "if " z >= 0) \ vb(E)_U &= vb(0) $
    //])
    where $x,y,z$ are coordinates in the *Lab* frame.
  ]
  #pause
  #block(width: 100%)[
    Transformation to the bunch frame is done as follows:
    $ vb(E') &= gamma(vb(E) + vb(v) times vb(B)) - (gamma - 1)(vb(E) dot vb(accent(v, hat)))vb(accent(v, hat)) \ 
      vb(B') &= gamma(vb(B) - (vb(v) times vb(E))/c^2) - (gamma - 1)(vb(B) dot vb(accent(v, hat)))vb(accent(v, hat)) $
    And since $c = 1$, $v = beta$ and $(vb(v) times vb(B))/c^2 = beta times vb(B)$
  ]
]
#normalislide(title: "Transformed Fields")[
  #set text(17pt)
  #block(width: 100%)[
    Insert $ z_"frame" = gamma_0(z_"bunch" - beta_0 t_"bunch") $
    into the undulator field
    #let zcord = $z_"frame"$
    $ vb(B)_U &= cases(vec(0, B_0 cosh(k_u y) k_u zcord e^(-(k_u zcord)^2/2), B_0 sinh(k_u y) e^(-(k_u zcord)^2/2)) "if" zcord < 0, vec(0, B_0 cosh(k_u y) sin(k_u zcord),B_0 sinh(k_u y) cos(k_u zcord)) "if " zcord >= 0) \ vb(E)_U &= vb(0) $
    Transform these fields with
    $ vb(E') &= gamma(vb(E) + vb(v) times vb(B)) - (gamma - 1)(vb(E) dot vb(accent(v, hat)))vb(accent(v, hat)) \ 
      vb(B') &= gamma(vb(B) - (vb(v) times vb(E))/c^2) - (gamma - 1)(vb(B) dot vb(accent(v, hat)))vb(accent(v, hat)) $
  ]
]
#normalislide(title: "Undulator parameter")[
  #set text(18pt)
  #block(width: 100%)[
    Define the undulator parameter:
    $ K &= (e B_0 lambda_U) / (2 pi m_e c) \ <==> B_0 &= (2 pi m_e c K) / (e lambda_U) \ "where " e &= 8.539 * 10^(-2) q_P \ m_e &= 4.185 * 10^(-23) m_P \ c &= 1 v_P $
    Significance: $ gamma_"frame" &= gamma_"bunch" / sqrt(1 + K^2/2) \ "FEL-IR:" K &= 1.417 $
  ]
]
#normalislide(title: "Radiation Result")[
  #image("rad.png", width: 80%)
  Plotted quantity $ integral_"front plane" vb(E) times vb(B) dot dd(S) $
]
#normalislide(title: "Radiation Evaluation")[
  The poynting vector $ vb(E) times vb(B) $ should be Lorentz-Invariant. \
  #{
    for i in range(1,10){
      text()[#i]
    }
  }
]
#normalislide(title: "Trajectory Result")[
  //#show ()
  #block(width: 100%)[
    Frame moves with $gamma_0$\
    Bunch moves with $gamma$, where $gamma_0 = gamma / sqrt(1+K^2/2)$ therefore $gamma > gamma_0$

    Bunch in lab frame slows down from $gamma$ to $gamma_0$, in bunch frame to $beta_z = 0$

    #grid(columns:(2fr, 1fr))[
    #image("tplot.svg", width: 100%)
    ][
      #text(15pt)[
        Problem: Bunch doesn't slow down to $ beta_z = 0 $
        Where $$
      ]
    ]
  ]
  
]