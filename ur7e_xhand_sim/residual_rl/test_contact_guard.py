import pytest
from contact_guard import assess_contacts

def contact(force,a='table',b='ur7e_xhand'):
    return dict(entity_a=a,entity_b=b,normal_force_N=force)

@pytest.mark.parametrize('forces,stop',[([],False),([0],False),([1,2],False),([2,3],False),([3,3],True),([7.2576475,5.8998532,3.9574594],True)])
def test_table_aggregate_threshold(forces,stop):
    result=assess_contacts([contact(f) for f in forces],5)
    assert result['stop']==stop
    assert result['table_normal_force_sum_N']==sum(forces)

def test_only_robot_table_and_both_orders():
    assert not assess_contacts([contact(100,'table','cube')])['stop']
    assert assess_contacts([contact(6,'ur7e_xhand','table')])['stop']

def test_self_collision_independent_of_table_threshold():
    assert assess_contacts([contact(0,'ur7e_xhand','ur7e_xhand')],100)['stop']

@pytest.mark.parametrize('force',[float('nan'),float('inf'),-1])
def test_invalid_force_fails_closed(force):
    result=assess_contacts([contact(force)])
    assert result['stop'] and result['table_normal_force_sum_N'] is None

@pytest.mark.parametrize('limit',[float('nan'),float('inf'),-1,True])
def test_invalid_config(limit):
    with pytest.raises(ValueError):assess_contacts([],limit)

def test_zero_and_configured_higher_limit():
    assert assess_contacts([contact(.001)],0)['stop']
    assert not assess_contacts([contact(17.115)],20)['stop']
