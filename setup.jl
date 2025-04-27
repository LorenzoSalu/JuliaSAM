# Questo file si occupa di installare tutte le dipendenze necessarie per il progetto

using Pkg

# Attiva l'ambiente corrente
Pkg.activate(@__DIR__)

# Installa tutte le dipendenze
Pkg.instantiate()

# Verifica che tutto sia installato correttamente
println("Tutte le dipendenze sono state installate correttamente!") 