# README


To render into reveal.js slides: `pandoc --mathjax --citeproc -t revealjs -s pres.md -o pres.html`


Example settings to use

```  
  title: Applying Reinforcement Learning to Hamiltonian Engineering
  author: Will Kaufman
  date: May 24, 2021
  
  bibliography: ../refs.bib
  
  theme: white
  transition: none
  slideNumber: true
  progress: false
  width: 1920
  height: 1080
```

Add this to remove text transform

```
<style>
.reveal h1, .reveal h2, .reveal h3, .reveal h4, .reveal h5 {
  text-transform: none;
}
</style>
```
