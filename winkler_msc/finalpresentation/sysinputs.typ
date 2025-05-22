#let fstate = "Gulliver"
#set par(justify: true)
#if sys.inputs.keys().contains("font"){
  fstate = sys.inputs.at("font")
}
#set text(1.3em, font: fstate)
#lorem(1000)