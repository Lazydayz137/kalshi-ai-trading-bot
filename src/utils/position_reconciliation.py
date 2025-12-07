"""
Position Reconciliation System
Ensures database state matches Kalshi API state to prevent discrepancies.
"""

import asyncio
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from dataclasses import dataclass

from src.clients.kalshi_client import KalshiClient
from src.utils.database import DatabaseManager, Position
from src.utils.logging_setup import get_trading_logger


@dataclass
class ReconciliationResult:
    """Result of position reconciliation."""
    timestamp: datetime
    kalshi_positions: int
    db_positions: int
    matched: int
    missing_in_db: int
    missing_in_kalshi: int
    discrepancies: List[Dict]
    success: bool


class PositionReconciliationSystem:
    """
    Reconciles positions between database and Kalshi API.

    Detects and resolves:
    - Positions in Kalshi but not in DB
    - Positions in DB but not in Kalshi (likely closed)
    - Price/quantity discrepancies
    - Stale position data
    """

    def __init__(
        self,
        db_manager: DatabaseManager,
        kalshi_client: KalshiClient
    ):
        self.db_manager = db_manager
        self.kalshi_client = kalshi_client
        self.logger = get_trading_logger("position_reconciliation")

    async def reconcile_positions(
        self,
        auto_fix: bool = True
    ) -> ReconciliationResult:
        """
        Reconcile positions between database and Kalshi API.

        Args:
            auto_fix: Whether to automatically fix discrepancies

        Returns:
            ReconciliationResult with details
        """
        self.logger.info("Starting position reconciliation")

        try:
            # Get positions from both sources
            kalshi_positions = await self._fetch_kalshi_positions()
            db_positions = await self._fetch_db_positions()

            # Create lookup maps
            kalshi_map = {p['market_id']: p for p in kalshi_positions}
            db_map = {p.market_id: p for p in db_positions}

            # Find discrepancies
            discrepancies = []
            matched = 0
            missing_in_db = 0
            missing_in_kalshi = 0

            # Check for positions in Kalshi but not in DB
            for market_id, kalshi_pos in kalshi_map.items():
                if market_id not in db_map:
                    missing_in_db += 1
                    discrepancy = {
                        'type': 'missing_in_db',
                        'market_id': market_id,
                        'kalshi_data': kalshi_pos,
                        'db_data': None
                    }
                    discrepancies.append(discrepancy)

                    if auto_fix:
                        await self._add_missing_position_to_db(kalshi_pos)
                else:
                    # Check for data discrepancies
                    db_pos = db_map[market_id]
                    if self._has_discrepancy(kalshi_pos, db_pos):
                        discrepancy = {
                            'type': 'data_mismatch',
                            'market_id': market_id,
                            'kalshi_data': kalshi_pos,
                            'db_data': db_pos
                        }
                        discrepancies.append(discrepancy)

                        if auto_fix:
                            await self._update_position_in_db(kalshi_pos, db_pos)
                    else:
                        matched += 1

            # Check for positions in DB but not in Kalshi (likely closed)
            for market_id, db_pos in db_map.items():
                if market_id not in kalshi_map:
                    missing_in_kalshi += 1
                    discrepancy = {
                        'type': 'missing_in_kalshi',
                        'market_id': market_id,
                        'kalshi_data': None,
                        'db_data': db_pos
                    }
                    discrepancies.append(discrepancy)

                    if auto_fix:
                        await self._handle_closed_position(db_pos)

            # Create result
            result = ReconciliationResult(
                timestamp=datetime.now(),
                kalshi_positions=len(kalshi_positions),
                db_positions=len(db_positions),
                matched=matched,
                missing_in_db=missing_in_db,
                missing_in_kalshi=missing_in_kalshi,
                discrepancies=discrepancies,
                success=len(discrepancies) == 0 or auto_fix
            )

            # Log results
            self.logger.info(
                "Position reconciliation complete",
                kalshi_positions=result.kalshi_positions,
                db_positions=result.db_positions,
                matched=result.matched,
                missing_in_db=result.missing_in_db,
                missing_in_kalshi=result.missing_in_kalshi,
                total_discrepancies=len(discrepancies),
                auto_fixed=auto_fix
            )

            if discrepancies:
                self.logger.warning(
                    f"Found {len(discrepancies)} position discrepancies",
                    discrepancies=discrepancies[:5]  # Log first 5
                )

            return result

        except Exception as e:
            self.logger.error(f"Position reconciliation failed: {e}")
            return ReconciliationResult(
                timestamp=datetime.now(),
                kalshi_positions=0,
                db_positions=0,
                matched=0,
                missing_in_db=0,
                missing_in_kalshi=0,
                discrepancies=[],
                success=False
            )

    async def _fetch_kalshi_positions(self) -> List[Dict]:
        """Fetch all positions from Kalshi API."""
        try:
            response = await self.kalshi_client.get_positions()
            positions = response.get('market_positions', [])

            # Convert to standardized format
            standardized = []
            for pos in positions:
                standardized.append({
                    'market_id': pos.get('ticker', ''),
                    'side': 'YES' if pos.get('position', 0) > 0 else 'NO',
                    'quantity': abs(pos.get('position', 0)),
                    'entry_price': pos.get('market_price', 0) / 100.0,  # Convert cents to dollars
                    'current_value': pos.get('total_traded', 0) / 100.0,
                })

            return standardized

        except Exception as e:
            self.logger.error(f"Failed to fetch Kalshi positions: {e}")
            return []

    async def _fetch_db_positions(self) -> List[Position]:
        """Fetch all open positions from database."""
        try:
            import aiosqlite

            async with aiosqlite.connect(self.db_manager.db_path) as db:
                cursor = await db.execute("""
                    SELECT market_id, side, entry_price, quantity, timestamp,
                           rationale, confidence, live, status, id, strategy
                    FROM positions
                    WHERE status = 'open'
                    ORDER BY timestamp DESC
                """)

                rows = await cursor.fetchall()

                positions = []
                for row in rows:
                    positions.append(Position(
                        market_id=row[0],
                        side=row[1],
                        entry_price=row[2],
                        quantity=row[3],
                        timestamp=datetime.fromisoformat(row[4]) if isinstance(row[4], str) else row[4],
                        rationale=row[5],
                        confidence=row[6],
                        live=bool(row[7]),
                        status=row[8],
                        id=row[9],
                        strategy=row[10]
                    ))

                return positions

        except Exception as e:
            self.logger.error(f"Failed to fetch DB positions: {e}")
            return []

    def _has_discrepancy(
        self,
        kalshi_pos: Dict,
        db_pos: Position
    ) -> bool:
        """Check if there's a discrepancy between Kalshi and DB position."""
        # Check quantity mismatch
        if abs(kalshi_pos['quantity'] - db_pos.quantity) > 0:
            return True

        # Check side mismatch
        if kalshi_pos['side'] != db_pos.side:
            return True

        # Price discrepancy is informational only (market price changes)
        # Don't treat as error

        return False

    async def _add_missing_position_to_db(self, kalshi_pos: Dict):
        """Add a position that exists in Kalshi but not in DB."""
        try:
            position = Position(
                market_id=kalshi_pos['market_id'],
                side=kalshi_pos['side'],
                entry_price=kalshi_pos['entry_price'],
                quantity=kalshi_pos['quantity'],
                timestamp=datetime.now(),
                rationale="Position recovered from Kalshi during reconciliation",
                confidence=None,
                live=True,
                status="open",
                strategy="reconciliation_recovery"
            )

            await self.db_manager.add_position(position)

            self.logger.info(
                f"Added missing position to DB: {kalshi_pos['market_id']}",
                quantity=kalshi_pos['quantity'],
                side=kalshi_pos['side']
            )

        except Exception as e:
            self.logger.error(
                f"Failed to add missing position {kalshi_pos['market_id']}: {e}"
            )

    async def _update_position_in_db(
        self,
        kalshi_pos: Dict,
        db_pos: Position
    ):
        """Update DB position to match Kalshi data."""
        try:
            import aiosqlite

            async with aiosqlite.connect(self.db_manager.db_path) as db:
                await db.execute("""
                    UPDATE positions
                    SET quantity = ?, side = ?
                    WHERE id = ?
                """, (kalshi_pos['quantity'], kalshi_pos['side'], db_pos.id))

                await db.commit()

            self.logger.info(
                f"Updated position in DB: {kalshi_pos['market_id']}",
                old_quantity=db_pos.quantity,
                new_quantity=kalshi_pos['quantity'],
                old_side=db_pos.side,
                new_side=kalshi_pos['side']
            )

        except Exception as e:
            self.logger.error(
                f"Failed to update position {kalshi_pos['market_id']}: {e}"
            )

    async def _handle_closed_position(self, db_pos: Position):
        """Handle a position that exists in DB but not in Kalshi (closed)."""
        try:
            # Mark as closed in database
            import aiosqlite

            async with aiosqlite.connect(self.db_manager.db_path) as db:
                await db.execute("""
                    UPDATE positions
                    SET status = 'closed'
                    WHERE id = ?
                """, (db_pos.id,))

                await db.commit()

            self.logger.info(
                f"Marked position as closed: {db_pos.market_id}",
                quantity=db_pos.quantity,
                side=db_pos.side
            )

            # Optionally create a trade log entry
            # (Would need to fetch fill data from Kalshi to get exit price)

        except Exception as e:
            self.logger.error(
                f"Failed to handle closed position {db_pos.market_id}: {e}"
            )


async def run_reconciliation_check(
    db_manager: DatabaseManager,
    kalshi_client: KalshiClient,
    auto_fix: bool = True
) -> ReconciliationResult:
    """
    Standalone function to run position reconciliation.

    Args:
        db_manager: Database manager instance
        kalshi_client: Kalshi client instance
        auto_fix: Whether to auto-fix discrepancies

    Returns:
        ReconciliationResult
    """
    reconciler = PositionReconciliationSystem(db_manager, kalshi_client)
    return await reconciler.reconcile_positions(auto_fix=auto_fix)
