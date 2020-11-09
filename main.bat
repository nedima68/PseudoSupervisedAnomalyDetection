python FabricDefectDetection.py dataset_name', type=click.Choice(['custom', 'AITEX']))
@click.argument('net_name', type=click.Choice(['FabricDefectDet_LeNetRELU']))
@click.argument('xp_path', type=click.Path(exists=True))
@click.argument('data_path', type=click.Path(exists=True))