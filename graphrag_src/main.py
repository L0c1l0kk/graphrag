import asyncio

from GraphGenerator import GraphGenerator

generator = GraphGenerator()
asyncio.run(generator.generate_graph())