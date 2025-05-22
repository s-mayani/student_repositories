// Get Polylux from the official package repository
#import "@preview/polylux:0.3.1": *
#import "@preview/physica:0.9.2": *
#import "@preview/pinit:0.1.3": *
#import "@preview/cetz:0.2.0": canvas, plot
//#import "tests.typ"
//#import themes.simple: *
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
#let titelsleid(title_image_height : 48%,showimage: false, author: "John Doe", email : "", coauthors : (), title : "Science", groupname : "SomeGroup") = {polylux-slide[
  #place(dx: 0%, dy: 6%, rect(stroke: 0pt, fill: rgb("dddddd"), width: 28%, height: title_image_height)[
    #place(dy: 7%)[#block(width: 100%, height: 100%, align(top + center, image("logos/PSI.png", width: 60%)))]
    #place(dy: 40%)[#block(width: 100%, height: 100%, align(top + center, image("logos/eth.png", width: 60%)))]
  ])
  #place(dx: 90%, dy: 6%, rect(stroke: 0pt, fill: rgb("dddddd"), width: 10%, height: title_image_height))
  #if showimage [
    #place(dx: 30%, dy: 6%, box(height: title_image_height)[
      #let tv = text(size: 15pt, "WIR SCHAFFEN WISSEN - HEUTE FÜR MORGEN")
      #image("logos/PSI_helicopter.jpg", height: 100%)
      #align(bottom + right, box(inset: 2pt, fill: rgb("cccccccc"), tv))
    ])
  ] else [
    #place(dx: 30%, dy: 6%, box(height: title_image_height)[
      #let tv = text(size: 15pt, "WIR SCHAFFEN WISSEN - HEUTE FÜR MORGEN")
      #rect(height: 100%, width: 15cm)
      #align(bottom + right, box(inset: 2pt, fill: rgb("cccccccc"), tv))
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
#titelsleid(title_image_height: 50%, author: "Manuel Winkler", email: "flaschenholz@protonmail.com", coauthors : ("S. Mayani", "A. Adelmann"), title : "Particles in IPPL", groupname : "AMAS Group, LSM")
#normalislide(title: "Outline")[
  #cool-outline(spacing : 10pt)
]
#newsectionslide(title: "Analytical Derivation")[]
#normalislide(title : "Math")[
  Some math equation: $ c = sqrt(a^2 + b^2) $
]
#newsectionslide(title: "Section Two")[]
#normalislide(title: "Citing papers")[
  @gonoskov_2022 and @Mur1981
]
#newsectionslide_nocontents(title: "Bibliography")[
  #set text(size: 14pt)
  #bibliography(title: "", "references.bib", style: "association-for-computing-machinery", full: false)
]