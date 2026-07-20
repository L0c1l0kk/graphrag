import asyncio
import logging
import time
 
from GraphGenerator import GraphGenerator

# Configure logging at entry point
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
 
 
generator = GraphGenerator(max_concurrent=4, description_model_name="forced-gpu-model")
start = time.perf_counter()
asyncio.run(generator.generate_graph())
elapsed = time.perf_counter() - start
hours, remainder = divmod(elapsed, 3600)
minutes, seconds = divmod(remainder, 60)
print(f"Total wall time: {int(hours)}h {int(minutes)}m {seconds:.1f}s")