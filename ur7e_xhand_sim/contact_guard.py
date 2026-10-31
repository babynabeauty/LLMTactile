"""Force-gated table protection for online simulation skills (not hardware)."""
import math

DEFAULT_TABLE_CONTACT_FORCE_LIMIT_N = 5.0


def validate_table_force_limit(value):
    if isinstance(value, bool) or not math.isfinite(value) or value < 0:
        raise ValueError('Table contact force limit must be finite and nonnegative')
    return float(value)


def assess_contacts(contacts, limit_N=DEFAULT_TABLE_CONTACT_FORCE_LIMIT_N):
    limit = validate_table_force_limit(limit_N)
    table, self_contacts = [], []
    for c in contacts:
        a, b = c['entity_a'], c['entity_b']
        if a == b == 'ur7e_xhand':
            self_contacts.append(c)
        elif {a, b} == {'ur7e_xhand', 'table'}:
            table.append(c)
    values = [float(c.get('normal_force_N', float('nan'))) for c in table]
    valid = all(math.isfinite(v) and v >= 0 for v in values)
    total = sum(values) if valid else None
    reason = None
    if self_contacts:
        reason = 'Robot self collision'
    elif not valid:
        reason = 'Invalid robot/table normal force; protection stop'
    elif total > limit:
        reason = f'Robot/table normal load {total:.6f} N exceeds limit {limit:g} N'
    return dict(table_normal_force_sum_N=total, table_force_limit_N=limit,
                table_contact_count=len(table), self_contact_count=len(self_contacts),
                force_valid=valid, stop=reason is not None, reason=reason,
                metric='sum of nonnegative solver normal forces across robot/table contacts; strict greater-than')
