import benchmark_models_petab
import pypesto
import pypesto.petab
import pypesto.optimize as opt
from pypesto.optimize.ess import SacessOptimizer
import pypesto.store

petab_problem = benchmark_models_petab.get_problem("Boehm_JProteomeRes2014")
pypesto_importer = pypesto.petab.PetabImporter(petab_problem)
pypesto_problem = pypesto_importer.create_problem()

sacess_options = opt.ess.get_default_ess_options(
    num_workers=12,
    dim=pypesto_problem.dim,
)
sacess = SacessOptimizer(ess_init_args=sacess_options, max_walltime_s=5e4)
result = sacess.minimize(pypesto_problem)

pypesto.store.write_result(
    result=result,
    filename="result.hdf5",
    optimize=True,
)
